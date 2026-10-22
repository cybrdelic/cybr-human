cmake_minimum_required(VERSION 3.20)
project(cybr_human_rasterizer LANGUAGES CXX)
add_library(rasterizer SHARED rasterizer.cpp)
target_compile_features(rasterizer PRIVATE cxx_std_17)
set_target_properties(rasterizer PROPERTIES
    PREFIX ""
    WINDOWS_EXPORT_ALL_SYMBOLS ON
    POSITION_INDEPENDENT_CODE ON
    CXX_EXTENSIONS OFF
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)
foreach(config Debug Release RelWithDebInfo MinSizeRel)
    string(TOUPPER "${config}" config_upper)
    set_target_properties(rasterizer PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}/lib"
        RUNTIME_OUTPUT_DIRECTORY_${config_upper} "${CMAKE_BINARY_DIR}/lib"
    )
endforeach()
