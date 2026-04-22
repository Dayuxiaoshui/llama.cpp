if (NOT MACA_PATH)
    if (NOT EXISTS $ENV{MACA_PATH})
        if (EXISTS /opt/maca)
            set(MACA_PATH /opt/maca)
        else()
            set(MACA_PATH /usr/local/cuda)
        endif()
    else()
        set(MACA_PATH $ENV{MACA_PATH})
    endif()
endif()

if (NOT MACA_PATH)
    set(MACA_PATH /usr/local/cuda)
endif()

if (DEFINED ENV{MACA_PATH} AND NOT MACA_PATH)
    set(MACA_PATH $ENV{MACA_PATH})
endif()

if (NOT EXISTS ${MACA_PATH})
    message(FATAL_ERROR "MetaX backend: MACA_PATH does not exist: ${MACA_PATH}")
endif()

if (NOT EXISTS "${MACA_PATH}/bin/nvcc")
    message(FATAL_ERROR "MetaX backend: expected CUDA-compat compiler at ${MACA_PATH}/bin/nvcc")
endif()

message(STATUS "MetaX backend: using MACA path ${MACA_PATH}")

set(METAX_NVCC "${MACA_PATH}/bin/nvcc")
if (EXISTS "/opt/maca/include")
    set(METAX_RUNTIME_PATH "/opt/maca")
else()
    set(METAX_RUNTIME_PATH "${MACA_PATH}")
endif()
if (EXISTS "/opt/maca/tools/cu-bridge/bin/gomxccbin")
    set(METAX_CUCC_PATH "/opt/maca/tools/cu-bridge")
elseif (EXISTS "${MACA_PATH}/tools/cu-bridge/bin/gomxccbin")
    set(METAX_CUCC_PATH "${MACA_PATH}/tools/cu-bridge")
else()
    message(FATAL_ERROR "MetaX backend: cannot locate cu-bridge (gomxccbin)")
endif()
set(METAX_CUBRIDGE_INCLUDE "${METAX_CUCC_PATH}/include")

file(GLOB GGML_HEADERS_METAX "../ggml-cuda/*.cuh")
list(APPEND GGML_HEADERS_METAX "../../include/ggml-cuda.h")
list(APPEND GGML_HEADERS_METAX "../ggml-metax/ggml-metax-compat.cuh")

file(GLOB GGML_SOURCES_METAX "../ggml-cuda/*.cu")
list(APPEND GGML_SOURCES_METAX "../ggml-metax/ggml-metax-fattn-stubs.cu")
list(APPEND GGML_SOURCES_METAX "../ggml-metax/quantize.cu")
# Start from a conservative kernel set to get MetaX backend compiling.
# We intentionally exclude CUDA-specific MMA/FATTN/MMQ paths first and
# can incrementally re-enable with MetaX-native implementations later.
list(FILTER GGML_SOURCES_METAX EXCLUDE REGEX ".*/ggml-cuda/fattn.*\\.cu$")
list(FILTER GGML_SOURCES_METAX EXCLUDE REGEX ".*/ggml-cuda/mmq.*\\.cu$")
list(FILTER GGML_SOURCES_METAX EXCLUDE REGEX ".*/ggml-cuda/mmf.*\\.cu$")
list(FILTER GGML_SOURCES_METAX EXCLUDE REGEX ".*/ggml-cuda/quantize\\.cu$")

set(METAX_NVCC_FLAGS
    -x cu
    --maca-path=${METAX_RUNTIME_PATH}
    -Xcompiler=-fPIC
    -std=c++17
    -use_fast_math
    -extended-lambda
    -imacros
    ${METAX_CUBRIDGE_INCLUDE}/__macro_mxcc.h
    -I${METAX_CUBRIDGE_INCLUDE}
    -I${MACA_PATH}/include
    -I${CMAKE_CURRENT_SOURCE_DIR}
    -I${CMAKE_CURRENT_SOURCE_DIR}/..
    -I${CMAKE_CURRENT_SOURCE_DIR}/../..
    -I${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

set(METAX_DEFS
    -DGGML_USE_METAX
    -DGGML_CUDA_PEER_MAX_BATCH_SIZE=${GGML_CUDA_PEER_MAX_BATCH_SIZE}
    -DGGML_CUDA_FORCE_CUBLAS
    -DGGML_CUDA_NO_FA
)
if (GGML_CUDA_GRAPHS)
    list(APPEND METAX_DEFS -DGGML_CUDA_USE_GRAPHS)
endif()
if (GGML_CUDA_FORCE_MMQ)
    list(APPEND METAX_DEFS -DGGML_CUDA_FORCE_MMQ)
endif()
if (GGML_CUDA_NO_VMM)
    list(APPEND METAX_DEFS -DGGML_CUDA_NO_VMM)
endif()
if (GGML_CUDA_NO_PEER_COPY)
    list(APPEND METAX_DEFS -DGGML_CUDA_NO_PEER_COPY)
endif()

set(METAX_OBJECTS)
foreach(SOURCE ${GGML_SOURCES_METAX})
    get_filename_component(SOURCE_ABS "${SOURCE}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    file(RELATIVE_PATH SOURCE_REL "${CMAKE_CURRENT_SOURCE_DIR}" "${SOURCE_ABS}")
    string(REPLACE "/" "_" OBJ_BASENAME "${SOURCE_REL}")
    string(REPLACE "." "_" OBJ_BASENAME "${OBJ_BASENAME}")

    # Patch sources to use MetaX-specific headers that avoid mma.cuh:
    if(SOURCE_REL MATCHES "ggml-cuda\\.cu$|mmvq\\.cu$")
        set(PATCHED_CU "${CMAKE_CURRENT_BINARY_DIR}/${OBJ_BASENAME}-patched.cu")
        file(READ "${SOURCE_ABS}" CU_CONTENT)
        string(REPLACE "#include \"ggml-cuda/fattn.cuh\"" "#include \"ggml-metax/ggml-cuda/fattn.cuh\"" CU_CONTENT "${CU_CONTENT}")
        string(REPLACE "#include \"ggml-cuda/mmq.cuh\""    "#include \"ggml-metax/ggml-cuda/mmq.cuh\""    CU_CONTENT "${CU_CONTENT}")
        string(REPLACE "#include \"ggml-cuda/mmf.cuh\""    "#include \"ggml-metax/ggml-cuda/mmf.cuh\""    CU_CONTENT "${CU_CONTENT}")
        string(REPLACE "#include \"quantize.cuh\""          "#include \"ggml-metax/ggml-cuda/quantize.cuh\"" CU_CONTENT "${CU_CONTENT}")
        string(REPLACE "#include \"ggml-cuda/quantize.cuh\"" "#include \"ggml-metax/ggml-cuda/quantize.cuh\"" CU_CONTENT "${CU_CONTENT}")
        # Fix bare .cuh includes that rely on the source directory being in the include path:
        string(REGEX REPLACE "#include \"([a-zA-Z0-9_-]+\\.cuh)\"" "#include \"ggml-cuda/\\1\"" CU_CONTENT "${CU_CONTENT}")
        file(WRITE "${PATCHED_CU}" "${CU_CONTENT}")
        set(SOURCE_ABS "${PATCHED_CU}")
    endif()

    set(OBJ_FILE "${CMAKE_CURRENT_BINARY_DIR}/metax_objs/${OBJ_BASENAME}.o")

    add_custom_command(
        OUTPUT "${OBJ_FILE}"
        COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_CURRENT_BINARY_DIR}/metax_objs"
        COMMAND ${CMAKE_COMMAND} -E env
            "MACA_PATH=${MACA_PATH}"
            "CUCC_PATH=${METAX_CUCC_PATH}"
            "C_INCLUDE_PATH=${METAX_CUBRIDGE_INCLUDE}:${MACA_PATH}/include"
            "CPLUS_INCLUDE_PATH=${METAX_CUBRIDGE_INCLUDE}:${MACA_PATH}/include"
            "${METAX_NVCC}" ${METAX_NVCC_FLAGS} ${METAX_DEFS} -c "${SOURCE_ABS}" -o "${OBJ_FILE}"
        DEPENDS "${SOURCE_ABS}" ${GGML_HEADERS_METAX}
        VERBATIM
    )
    list(APPEND METAX_OBJECTS "${OBJ_FILE}")
endforeach()

ggml_add_backend_library(ggml-metax
                         ${GGML_HEADERS_METAX}
                         ${METAX_OBJECTS}
                        )

if (NOT GGML_BACKEND_DL)
    target_compile_definitions(ggml PUBLIC GGML_USE_CUDA)
endif()
target_include_directories(ggml-metax PRIVATE "${MACA_PATH}/include")

target_link_directories(ggml-metax PRIVATE "${MACA_PATH}/lib64")
if (GGML_STATIC)
    target_link_libraries(ggml-metax PRIVATE cudart_static mcblas_static mcblasLt_static)
else()
    target_link_libraries(ggml-metax PRIVATE cudart mcblas mcblasLt)
endif()

if (NOT GGML_CUDA_NO_VMM)
    # CUDA-compat driver runtime for virtual memory management APIs
    target_link_libraries(ggml-metax PRIVATE mxc-runtime64)
endif()

set_target_properties(ggml-metax PROPERTIES LINKER_LANGUAGE CXX)
