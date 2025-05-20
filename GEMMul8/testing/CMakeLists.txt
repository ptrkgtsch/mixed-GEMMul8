cmake_minimum_required(VERSION 3.18)
if (NOT CMAKE_GPU_RUNTIME)
    set(GPU_RUNTIME "CUDA" CACHE STRING "Switches between HIP and CUDA")
else (NOT CMAKE_GPU_RUNTIME)
    set(GPU_RUNTIME "${CMAKE_GPU_RUNTIME}" CACHE STRING "Switches between HIP and CUDA")
endif (NOT CMAKE_GPU_RUNTIME)

# project(TestGEMMul LANGUAGES CXX ${GPU_RUNTIME})

if (${GPU_RUNTIME} MATCHES "HIP")
    set (VECTORADD_CXX_FLAGS "-fPIC")
    find_package(hipblas REQUIRED)
    include_directories(${HIPBLAS_INCLUDE_DIRS})
    message(STATUS "HIPBLAS_INCLUDE_DIRS: ${HIPBLAS_INCLUDE_DIRS}")
    set(CMAKE_CXX_COMPILER hipcc)
elseif (${GPU_RUNTIME} MATCHES "CUDA")
    find_package(CUDA 11.0 REQUIRED)
    find_package(CUDAToolkit REQUIRED)
    set(CMAKE_CUDA_ARCHITECTURES 80)
else ()
    message (FATAL_ERROR "GPU runtime not supported!")
endif ()

# CUDA
set(CMAKE_${GPU_RUNTIME}_STANDARD 20)
set(CMAKE_${GPU_RUNTIME}_STANDARD_REQUIRED ON)
set(CMAKE_${GPU_RUNTIME}_EXTENSIONS OFF)

# Directories
set(INCDIR ../include)
set(OZIMMU_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../ozIMMU_EF")
set(CUMP_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../cuMpSGEMM")

# Options for ozIMMU_EF and cuMpSGEMM
option(USE_OZIMMU_EF "Use ozIMMU_EF" ON)
option(USE_CUMP "Use cuMpSGEMM" ON)

# Set NVCC flags
set(NVCC_FLAGS -O3)

# GPU architecture
if (${GPU_RUNTIME} MATCHES "CUDA")
    execute_process(COMMAND nvidia-smi --query-gpu=compute_cap --format=csv,noheader
                    OUTPUT_VARIABLE GPU_ARCH
                    OUTPUT_STRIP_TRAILING_WHITESPACE)
    string(REPLACE "." "" GPU_ARCH ${GPU_ARCH})  # Remove the dot from the architecture
    list(APPEND NVCC_FLAGS "-gencode" "arch=compute_${GPU_ARCH},code=sm_${GPU_ARCH}")
endif ()

# Include directories
include_directories(${INCDIR})

# Libraries
if (${GPU_RUNTIME} MATCHES "CUDA")
    set(LIBS -lnvidia-ml -lcublas -lcudart -lcuda -L../lib -lgemmul8)
else ()
    set(LIBS -lamdhip64 -lhipblas -L../lib -lgemmul8)
endif()

if (${GPU_RUNTIME} MATCHES "CUDA")
    if(USE_OZIMMU_EF)
        add_definitions(-DozIMMU_EF_FLAG)
        include_directories(${OZIMMU_DIR}/include)
        list(APPEND LIBS -L${OZIMMU_DIR}/build -lozimmu)
    endif()

    if(USE_CUMP)
        add_definitions(-DcuMpSGEMM_FLAG)
        include_directories(${CUMP_DIR}/include)
        list(APPEND LIBS -L${CUMP_DIR}/build -lcumpsgemm)
    endif()
endif ()

link_directories(${CMAKE_CURRENT_SOURCE_DIR}/../lib)  # Add the path to the gemmul8 library
if (${GPU_RUNTIME} MATCHES "CUDA")
    if(USE_OZIMMU_EF)
        link_directories(${OZIMMU_DIR}/build)  # Add the path to the ozIMMU_EF library
    endif()
    if(USE_CUMP)
        link_directories(${CUMP_DIR}/build)  # Add the path to the cuMpSGEMM library
    endif()
endif ()

# Create test executables
set(TEST_SOURCES
    test_double.cu
    test_float.cu
    test_mixed_double.cu
    test_mixed_float.cu
)

foreach(TEST_SOURCE ${TEST_SOURCES})
    if (${GPU_RUNTIME} MATCHES "HIP")
        set_source_files_properties(${TEST_SOURCE} PROPERTIES LANGUAGE HIP)
    endif ()
    get_filename_component(TEST_NAME ${TEST_SOURCE} NAME_WE)
    add_executable(${TEST_NAME} ${TEST_SOURCE})
    
    # Set NVCC flags for the target
    target_compile_options(${TEST_NAME} PRIVATE ${NVCC_FLAGS})
    
    # Link libraries
    target_link_libraries(${TEST_NAME} PRIVATE gemmul8_static ${LIBS})
    # Do not forget to copy the file to the right location
    add_dependencies(${TEST_NAME} copy_gemmul8)

    # Set the library path for runtime
    if (${GPU_RUNTIME} MATCHES "CUDA")
        set_target_properties(${TEST_NAME} PROPERTIES
            BUILD_WITH_INSTALL_RPATH TRUE
            INSTALL_RPATH "${CUDA_TOOLKIT_ROOT_DIR}/lib64;${OZIMMU_DIR}/build;${CUMP_DIR}/build"
        )
    endif ()
endforeach()

if (${GPU_RUNTIME} MATCHES "CUDA")
    ## Command to get the nvcc_version file
    execute_process(
            COMMAND nvcc --version
            OUTPUT_VARIABLE NVCC_VERSION_OUTPUT
    )

    # Write the output to a file
    file(WRITE "${CMAKE_BINARY_DIR}/nvcc_version" "${NVCC_VERSION_OUTPUT}")
endif ()

# Clean target
set_property(DIRECTORY APPEND PROPERTY ADDITIONAL_MAKE_CLEAN_FILES "${CMAKE_BINARY_DIR}/nvcc_version")
