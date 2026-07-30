/*
    Abaqus mesh IO helpers

    This header declares functions implemented in:
        file_io_abaqus.cpp

    Purpose:
      - Read Abaqus .inp mesh into your HexMesh container so the rest of your
        pipeline can remain the same as the COMSOL reader path.
*/

#ifndef file_io_abaqus_h
#define file_io_abaqus_h

#include <string>
#include "myMeshGenerator.h"   // for HexMesh

// Read Abaqus .inp mesh file and return a HexMesh.
// Implemented in file_io_abaqus.cpp
HexMesh readAbaqusInputO(const std::string& filename);

#endif // file_io_abaqus_h
