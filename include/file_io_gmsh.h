#ifndef file_io_gmsh_h
#define file_io_gmsh_h

#include <string>
#include "myMeshGenerator.h"   // for HexMesh

// Read Gmsh v2.2 ASCII .msh (tetra + optional triangles) and set boundary_flag using
// bottom/top node lists produced by your python script.
// - mshFile:                 path to "dura_partition_wound.msh"
// - bottomNodesTxt/topNodesTxt: paths to "bottom_end_nodes.txt" / "top_end_nodes.txt"

HexMesh readGmshMsh22(const std::string& mshFile,
                      const std::string& bottomNodesTxt,
                      const std::string& topNodesTxt);

HexMesh readGmshMsh22doubleWound(const std::string& mshFile,
                                const std::string& bottomNodesTxt,
                                const std::string& topNodesTxt,
                                int wound1_tag = 101,
                                int wound2_tag = 104,
                                int healthy_tag = 102);

#endif // 
