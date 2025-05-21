cmake_minimum_required(VERSION 3.18)
if (NOT CMAKE_GPU_RUNTIME)
    set(GPU_RUNTIME "CUDA" CACHE STRING "Switches between HIP and CUDA")
else (NOT CMAKE_GPU_RUNTIME)
    set(GPU_RUNTIME "${CMAKE_GPU_RUNTIME}" CACHE STRING "Switches between HIP and CUDA")
endif (NOT CMAKE_GPU_RUNTIME)

project(GEMMul8 LANGUAGES CXX ${GPU_RUNTIME})

if (${GPU_RUNTIME} MATCHES "HIP")
    set (VECTORADD_CXX_FLAGS "-fPIC")
    find_package(HIP 6.2.0 REQUIRED)
    find_package(hipblas REQUIRED)
    # add_compile_definitions(OCML_BASIC_ROUNDED_OPERATIONS) # for _ru and _rd modes (slow!)
    add_compile_definitions(HIP_ENABLE_WARP_SYNC_BUILTINS) # for __shfl_down_sync
    add_compile_options(-Wno-unused-result)
elseif (${GPU_RUNTIME} MATCHES "CUDA")
    find_package(CUDA 11.0 REQUIRED)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_ARCHITECTURES 80)
else ()
    message (FATAL_ERROR "GPU runtime not supported!")
endif ()

foreach(lang CXX ${GPU_RUNTIME})
    set(CMAKE_${lang}_STANDARD 20)
    set(CMAKE_${lang}_STANDARD_REQUIRED ON)
    set(CMAKE_${lang}_EXTENSIONS OFF)
endforeach()

# Directories
set(INCDIR include)
set(SRCDIR src)

# Header files
file(GLOB HEADERS "${INCDIR}/*.hpp")

# Source files
set(CU_FILES
    ${SRCDIR}/gemmul8.cu
)
if (${GPU_RUNTIME} MATCHES HIP)
    set_source_files_properties(${SRCDIR}/gemmul8.cu PROPERTIES LANGUAGE HIP)
endif ()

# Create an object library for the CUDA sources
add_library(gemmul8_objs OBJECT ${CU_FILES})
set_property(TARGET gemmul8_objs PROPERTY POSITION_INDEPENDENT_CODE ON)
target_include_directories(gemmul8_objs PUBLIC ${INCDIR})

# Create a static library from the object library
add_library(gemmul8_static STATIC $<TARGET_OBJECTS:gemmul8_objs>)
set_target_properties(gemmul8_static PROPERTIES OUTPUT_NAME "gemmul8")  # Rename output to gemmul8.a (use only if shared not included)
if (${GPU_RUNTIME} MATCHES "CUDA")
    target_link_libraries(gemmul8_static PRIVATE cublas cudart cuda)
else ()
    target_link_libraries(gemmul8_static ${HIP_LIBRARIES})
endif ()

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
add_custom_command(
    OUTPUT ${CMAKE_SOURCE_DIR}/lib/libgemmul8.a
    COMMAND ${CMAKE_COMMAND} -E copy $<TARGET_FILE:gemmul8_static> ${CMAKE_SOURCE_DIR}/lib/libgemmul8.a
        DEPENDS gemmul8_static
    COMMENT "Copying libgemmul8.a to ./lib/"
)

# Create a custom target that depends on the custom command
add_custom_target(copy_gemmul8 ALL DEPENDS ${CMAKE_SOURCE_DIR}/lib/libgemmul8.a)

# Ensure that the custom target is built whenever a target that links against gemmul8_static is built:
# Add `add_dependencies(my_target copy_gemmul8)`

# Add Testing to be able to run it
add_subdirectory("${PROJECT_SOURCE_DIR}/testing")
