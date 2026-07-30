// ------------------------------------------------------------
// Gmsh v2.2 ASCII reader for DOUBLE-WOUND dura mesh
// - Reads nodes + tetra (type 4)
// - Uses physical tags on tets as element_domain_id:
//     wound1_tag = 101, wound2_tag = 104, healthy_tag = 102 (you can change)
// - Uses python-produced node lists to:
//     (1) set myMesh.boundary_flag[node] = 1 for top/bottom nodes
//     (2) build cap triangle elements (top/bottom) from tet boundary faces
//         and store them in myMesh.surface_elements
//         with myMesh.surface_boundary_flag = 1 (bottom) or 2 (top)
//
// NOTE:
// This does NOT require Gmsh to export triangles. We reconstruct boundary faces
// from tets (faces that appear once).
// ------------------------------------------------------------

#include <fstream>
#include <sstream>
#include <unordered_set>
#include <unordered_map>
#include <stdexcept>
#include <vector>
#include <string>
#include <array>
#include <algorithm>
#include <iostream>
#include <Eigen/Dense>
#include "element_functions.h"
#include "solver.h"
#include "myMeshGenerator.h"

// helper: load node ids (one int per line) into set (0-based)
static std::unordered_set<int> loadNodeSet0Based(const std::string& filename)
{
    std::unordered_set<int> s;
    std::ifstream f(filename);
    if(!f.is_open())
        throw std::runtime_error("Failed to open node list: " + filename);

    int id;
    while(f >> id) s.insert(id);
    return s;
}

// ---- face key for hashing (sorted node ids) ----
struct FaceKey {
    int a, b, c; // always sorted a<b<c
    bool operator==(const FaceKey& o) const {
        return a==o.a && b==o.b && c==o.c;
    }
};

struct FaceKeyHash {
    std::size_t operator()(FaceKey const& k) const noexcept {
        // a simple but decent hash combine
        std::size_t h1 = std::hash<int>{}(k.a);
        std::size_t h2 = std::hash<int>{}(k.b);
        std::size_t h3 = std::hash<int>{}(k.c);
        std::size_t h = h1;
        h ^= (h2 + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2));
        h ^= (h3 + 0x9e3779b97f4a7c15ULL + (h<<6) + (h>>2));
        return h;
    }
};

static inline FaceKey makeFaceKey(int i0, int i1, int i2)
{
    std::array<int,3> a = {i0,i1,i2};
    std::sort(a.begin(), a.end());
    return FaceKey{a[0], a[1], a[2]};
}

// Build external boundary faces of tetra mesh:
// faces that occur exactly once among all tet faces.
static std::vector<std::array<int,3>> boundaryFacesFromTets(const std::vector<std::vector<int>>& tets)
{
    // count faces
    std::unordered_map<FaceKey, int, FaceKeyHash> faceCount;
    faceCount.reserve(tets.size() * 2);

    auto addFace = [&](int i0,int i1,int i2){
        FaceKey k = makeFaceKey(i0,i1,i2);
        auto it = faceCount.find(k);
        if(it == faceCount.end()) faceCount.emplace(k, 1);
        else it->second += 1;
    };

    for(const auto& e : tets)
    {
        // e has 4 node ids
        const int n0 = e[0], n1 = e[1], n2 = e[2], n3 = e[3];
        addFace(n0,n1,n2);
        addFace(n0,n1,n3);
        addFace(n0,n2,n3);
        addFace(n1,n2,n3);
    }

    // collect faces that occur once
    std::vector<std::array<int,3>> bfaces;
    bfaces.reserve(faceCount.size()/2);

    for(const auto& kv : faceCount)
    {
        if(kv.second == 1)
        {
            const FaceKey& k = kv.first;
            bfaces.push_back({k.a, k.b, k.c}); // sorted ids; orientation not needed for your bc
        }
    }
    return bfaces;
}

// Reads Gmsh v2.2 ASCII .msh with tetra physical tags,
// and reconstructs TOP/BOTTOM cap triangles from tet boundary faces.
HexMesh readGmshMsh22doubleWound(const std::string& mshFile,
                                const std::string& bottomNodesTxt,
                                const std::string& topNodesTxt,
                                int wound1_tag = 101,
                                int wound2_tag = 104,
                                int healthy_tag = 102)
{
    std::cout << "Importing DOUBLE-WOUND mesh from Gmsh .msh (v2.2 ASCII)\n";

    // load boundary node sets (top/bottom) from python (0-based)
    const auto bottomSet = loadNodeSet0Based(bottomNodesTxt);
    const auto topSet    = loadNodeSet0Based(topNodesTxt);

    std::ifstream in(mshFile);
    if(!in.is_open())
        throw std::runtime_error("Failed to open msh file: " + mshFile);

    std::vector<Eigen::Vector3d> NODES;
    std::vector<std::vector<int>> TETS;
    std::vector<int> TET_PHYS; // physical tag per tet (domain id)

    std::vector<int> BOUNDARIES;          // node boundary flag 0/1 (top or bottom)
    std::vector<std::vector<int>> TRIS;   // reconstructed cap triangles (top+bottom)
    std::vector<int> TRI_BC;              // 1 bottom, 2 top

    std::string line;

    auto expectLine = [&](const std::string& s){
        if(!std::getline(in, line))
            throw std::runtime_error("Unexpected EOF while looking for: " + s);
        if(line != s)
            throw std::runtime_error("Expected '" + s + "' but got '" + line + "'");
    };

    bool gotNodes = false;
    bool gotElems = false;

    while(std::getline(in, line))
    {
        // -----------------------
        // NODES
        // -----------------------
        if(line == "$Nodes")
        {
            gotNodes = true;
            std::getline(in, line);
            int nNodes = std::stoi(line);
            NODES.resize(nNodes);

            for(int i=0; i<nNodes; ++i)
            {
                std::getline(in, line);
                std::istringstream iss(line);
                int id1; double x,y,z;
                iss >> id1 >> x >> y >> z;

                // gmsh node ids are 1-based; assume contiguous 1..n
                int idx = id1 - 1;
                if(idx < 0 || idx >= nNodes)
                    throw std::runtime_error("Non-contiguous node ids in msh; need a map-based reader.");
                NODES[idx] = Eigen::Vector3d(x,y,z);
            }
            expectLine("$EndNodes");

            // boundary flags: mark top/bottom nodes
            BOUNDARIES.assign(nNodes, 0);
            for(int i=0; i<nNodes; ++i)
            {
                if(bottomSet.count(i) || topSet.count(i))
                    BOUNDARIES[i] = 1;
            }
        }

        // -----------------------
        // ELEMENTS
        // -----------------------
        if(line == "$Elements")
        {
            gotElems = true;
            std::getline(in, line);
            int nElem = std::stoi(line);

            TETS.reserve(nElem);
            TET_PHYS.reserve(nElem);

            for(int e=0; e<nElem; ++e)
            {
                std::getline(in, line);
                std::istringstream iss(line);

                int elmNumber, elmType, numTags;
                iss >> elmNumber >> elmType >> numTags;

                int physical = 0;
                for(int t=0; t<numTags; ++t)
                {
                    int tag; iss >> tag;
                    if(t == 0) physical = tag; // tag1 = physical group id
                }

                if(elmType == 4) // 4-node tetra
                {
                    int n0,n1,n2,n3;
                    iss >> n0 >> n1 >> n2 >> n3;
                    std::vector<int> conn = {n0-1, n1-1, n2-1, n3-1};
                    TETS.push_back(conn);
                    TET_PHYS.push_back(physical);
                }
                else
                {
                    // ignore: triangles/lines/points/etc.
                }
            }
            expectLine("$EndElements");
        }
    }

    if(!gotNodes) throw std::runtime_error("No $Nodes section found in msh.");
    if(!gotElems) throw std::runtime_error("No $Elements section found in msh.");
    if(TETS.empty()) throw std::runtime_error("No tetra elements found in msh.");

    // ------------------------------------------------------------
    // Reconstruct boundary faces from tets, then classify caps:
    // - bottom face: all 3 nodes in bottomSet
    // - top face:    all 3 nodes in topSet
    //
    // Store ONLY cap faces in TRIS, with TRI_BC = 1 or 2.
    // ------------------------------------------------------------
    {
        std::vector<std::array<int,3>> bfaces = boundaryFacesFromTets(TETS);
        TRIS.reserve(bfaces.size()/4);
        TRI_BC.reserve(bfaces.size()/4);

        for(const auto& f : bfaces)
        {
            const int a = f[0], b = f[1], c = f[2];
            const bool isBottom = bottomSet.count(a) && bottomSet.count(b) && bottomSet.count(c);
            const bool isTop    = topSet.count(a)    && topSet.count(b)    && topSet.count(c);

            if(isBottom || isTop)
            {
                TRIS.push_back({a,b,c});
                TRI_BC.push_back(isBottom ? 1 : 2);
            }
        }

        std::cout << "Reconstructed boundary faces: " << bfaces.size() << "\n";
        std::cout << "Cap triangles stored: " << TRIS.size()
                  << " (bc_id 1=bottom, 2=top)\n";
        if(TRIS.empty())
        {
            std::cout << "WARNING: No cap triangles detected from node sets.\n"
                      << "Check that your bottom/top node lists match the mesh indexing.\n";
        }
    }

    // ------------------------------------------------------------
    // Build mesh object
    // ------------------------------------------------------------
    HexMesh myMesh;
    myMesh.nodes = std::move(NODES);
    myMesh.elements = std::move(TETS);
    myMesh.boundary_flag = std::move(BOUNDARIES);

    myMesh.surface_elements = std::move(TRIS);
    myMesh.surface_boundary_flag = std::move(TRI_BC);

    myMesh.element_domain_id = std::move(TET_PHYS);

    myMesh.n_nodes = (int)myMesh.nodes.size();
    myMesh.n_elements = (int)myMesh.elements.size();
    myMesh.n_surf_elements = (int)myMesh.surface_elements.size();

    // quick sanity counts of domain ids
    int c_w1=0, c_w2=0, c_h=0, c_other=0;
    for(int dom : myMesh.element_domain_id){
        if(dom == wound1_tag) ++c_w1;
        else if(dom == wound2_tag) ++c_w2;
        else if(dom == healthy_tag) ++c_h;
        else ++c_other;
    }

    std::cout << "Created mesh with "
              << myMesh.n_nodes << " nodes, "
              << myMesh.n_elements << " tetra elements.\n";
    std::cout << "Domain counts: wound1(" << wound1_tag << ")=" << c_w1
              << ", wound2(" << wound2_tag << ")=" << c_w2
              << ", healthy(" << healthy_tag << ")=" << c_h
              << ", other=" << c_other << "\n";
    std::cout << "Stored " << myMesh.n_surf_elements
              << " cap triangles in surface_elements.\n";

    return myMesh;
}
