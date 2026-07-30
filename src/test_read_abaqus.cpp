#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <unordered_map>
#include <limits>

#include "file_io_abaqus.h"   // declares HexMesh readAbaqusInput(...)
#include "myMeshGenerator.h"  // HexMesh definition (usually already pulled by header)

int main(int argc, char** argv)
{
    try {
        // ---- Choose input file ----
        // Usage:
        //   ./test_read_abaqus your_mesh.inp
        //
        // If not provided, use a default name (edit this as needed).
        std::string filename;
        if (argc >= 2) {
            filename = argv[1];
        } else {
            filename = "square_mechanics_patchSize_01_C3D8_meshSize_003_patch_global_02_full.inp";
        }

        std::cout << "Reading Abaqus mesh: " << filename << "\n";

        // ---- Read mesh ----
        HexMesh mesh = readAbaqusInputO(filename);

        // ---- Basic stats ----
        std::cout << "Mesh stats:\n";
        std::cout << "  n_nodes        = " << mesh.n_nodes << "\n";
        std::cout << "  n_elements     = " << mesh.n_elements << "\n";
        std::cout << "  n_surf_elements= " << mesh.n_surf_elements << "\n";

        // ---- Bounding box ----
        double xmin =  std::numeric_limits<double>::infinity();
        double ymin =  std::numeric_limits<double>::infinity();
        double zmin =  std::numeric_limits<double>::infinity();
        double xmax = -std::numeric_limits<double>::infinity();
        double ymax = -std::numeric_limits<double>::infinity();
        double zmax = -std::numeric_limits<double>::infinity();

        for (const auto& p : mesh.nodes) {
            xmin = std::min(xmin, p(0)); xmax = std::max(xmax, p(0));
            ymin = std::min(ymin, p(1)); ymax = std::max(ymax, p(1));
            zmin = std::min(zmin, p(2)); zmax = std::max(zmax, p(2));
        }

        std::cout << std::setprecision(8);
        std::cout << "Bounding box:\n";
        std::cout << "  X: [" << xmin << ", " << xmax << "]\n";
        std::cout << "  Y: [" << ymin << ", " << ymax << "]\n";
        std::cout << "  Z: [" << zmin << ", " << zmax << "]\n";

        // ---- Boundary node count (NSET 'bc' -> boundary_flag==1) ----
        int n_bc = 0;
        for (int i = 0; i < (int)mesh.boundary_flag.size(); ++i) {
            if (mesh.boundary_flag[i] == 1) n_bc++;
        }
        std::cout << "Boundary nodes (boundary_flag==1): " << n_bc << "\n";

        // ---- Print first node ----
        if (!mesh.nodes.empty()) {
            std::cout << "First node (index 0): " << mesh.nodes[0].transpose() << "\n";
        }

        // ---- Print first element connectivity ----
        if (!mesh.elements.empty()) {
            std::cout << "First C3D8 element connectivity (0-based node indices): ";
            for (int j = 0; j < (int)mesh.elements[0].size(); ++j) {
                std::cout << mesh.elements[0][j] << (j+1==(int)mesh.elements[0].size() ? "" : " ");
            }
            std::cout << "\n";
        }

        // ---- Optional: Surface tag histogram ----
        // surface_boundary_flag should be same length as surface_elements
        if (!mesh.surface_boundary_flag.empty()) {
            std::unordered_map<int,int> tag_counts;
            for (int tag : mesh.surface_boundary_flag) tag_counts[tag]++;

            std::cout << "Surface tag counts:\n";
            for (auto& kv : tag_counts) {
                std::cout << "  tag " << kv.first << ": " << kv.second << " quad faces\n";
            }
        } else {
            std::cout << "No surface faces found (no *Surface in inp, or not parsed).\n";
        }

        std::cout << "Done.\n";
        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 1;
    }
}
