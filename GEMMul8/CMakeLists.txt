cmake_minimum_required(VERSION 3.18)
project(GEMMul8 LANGUAGES CXX CUDA)

find_package(CUDA 11.0 REQUIRED)
find_package(CUDAToolkit REQUIRED)

# CUDA/CXX
foreach(lang CXX CUDA)
    set(CMAKE_${lang}_STANDARD 20)
    set(CMAKE_${lang}_STANDARD_REQUIRED ON)
    set(CMAKE_${lang}_EXTENSIONS OFF)
endforeach()

# CUDA (1/2)
set(CMAKE_CUDA_ARCHITECTURES 80)

# Directories
set(INCDIR include)
set(SRCDIR src)

# Header files
file(GLOB HEADERS "${INCDIR}/*.hpp")

# Source files
set(CU_FILES
    ${SRCDIR}/gemmul8.cu
)

# Create an object library for the CUDA sources
add_library(gemmul8_objs OBJECT ${CU_FILES})
set_property(TARGET gemmul8_objs PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(gemmul8_objs PUBLIC ${INCDIR})

# Create a static library from the object library
add_library(gemmul8_static STATIC $<TARGET_OBJECTS:gemmul8_objs>)
set_target_properties(gemmul8_static PROPERTIES OUTPUT_NAME "gemmul8")  # Rename output to gemmul8.a (use only if shared not included)
target_link_libraries(gemmul8_static PRIVATE cublas cudart cuda)

# Create a shared library from the object library
# add_library(gemmul8 SHARED $<TARGET_OBJECTS:gemmul8_objs>)
# target_link_libraries(gemmul8 PRIVATE cublas cudart cuda)

##########################################################################
# Installing
##########################################################################
install(TARGETS gemmul8_static
    ARCHIVE DESTINATION lib
    PUBLIC_HEADER DESTINATION include/gemmul8
 )

# install(TARGETS gemmul8
#     LIBRARY DESTINATION lib
# )

# Custom command to copy the static library to ./lib
add_custom_command(TARGET gemmul8_static POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:gemmul8_static> ${CMAKE_SOURCE_DIR}/lib/libgemmul8.a
        COMMENT "Copying libgemmul8.a to ./lib/"
)

# Add Testing to be able to run it
add_subdirectory("${PROJECT_SOURCE_DIR}/testing")
