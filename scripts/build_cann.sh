#!/usr/bin/env bash
set -euo pipefail
umask 022

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

CMAKE_BIN="${CMAKE_BIN:-cmake}"
BUILD_DIR="${BUILD_DIR:-build-cann}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
SOC_TYPE="${SOC_TYPE:-Ascend910C}"
USE_ACL_GRAPH="${USE_ACL_GRAPH:-OFF}"
ED_BENCHMARK_MARKERS="${ED_BENCHMARK_MARKERS:-ON}"

if [[ -z "${CANN_INSTALL_DIR:-}" ]]; then
  for candidate in \
    "${ASCEND_TOOLKIT_HOME:-}" \
    "${ASCEND_HOME_PATH:-}" \
    /usr/local/Ascend/ascend-toolkit/latest \
    /usr/local/Ascend/latest; do
    if [[ -n "${candidate}" && -f "${candidate}/include/acl/acl.h" && -f "${candidate}/lib64/libascendcl.so" ]]; then
      CANN_INSTALL_DIR="${candidate}"
      break
    fi
  done
fi

if [[ -z "${CANN_INSTALL_DIR:-}" ]]; then
  echo "error: CANN Toolkit was not found." >&2
  echo "hint: source the CANN set_env.sh script or set CANN_INSTALL_DIR explicitly." >&2
  exit 1
fi

if [[ -f "${CANN_INSTALL_DIR}/set_env.sh" ]]; then
  source "${CANN_INSTALL_DIR}/set_env.sh"
fi

if ! command -v "${CMAKE_BIN}" >/dev/null 2>&1 && [[ ! -x "${CMAKE_BIN}" ]]; then
  echo "error: CMake was not found. Install CMake or set CMAKE_BIN=/path/to/cmake." >&2
  exit 1
fi

if [[ ! -f third_party/ggml/CMakeLists.txt ]]; then
  echo "error: missing ggml submodule; run bash scripts/bootstrap.sh first." >&2
  exit 1
fi

if [[ "${CLEAN:-0}" == "1" ]]; then
  rm -rf -- "${BUILD_DIR}"
fi

if [[ -z "${JOBS:-}" ]]; then
  JOBS="$(nproc)"
  if (( JOBS > 32 )); then
    JOBS=32
  fi
fi

cat <<EOF
CANN build configuration:
  source:           ${ROOT_DIR}
  build:            ${BUILD_DIR}
  CANN Toolkit:     ${CANN_INSTALL_DIR}
  SoC:              ${SOC_TYPE}
  ACL graph:        ${USE_ACL_GRAPH}
  accelerator:      CANN=ON, CUDA/HIP/SYCL/Vulkan/Metal/OpenCL=OFF
  CUDA features:    cuDNN/NCCL/graphs/custom kernels=OFF
  parallel runtime: OFF
  build type:       ${BUILD_TYPE}
  parallel jobs:    ${JOBS}
EOF

"${CMAKE_BIN}" -S . -B "${BUILD_DIR}" \
  "-DCMAKE_BUILD_TYPE=${BUILD_TYPE}" \
  -DED_BUILD_EXAMPLES=ON \
  -DED_GGML_CANN=ON \
  -DGGML_CANN=ON \
  -DED_GGML_CUDA=OFF \
  -DGGML_CUDA=OFF \
  -DED_GGML_HIP=OFF \
  -DGGML_HIP=OFF \
  -DED_GGML_SYCL=OFF \
  -DGGML_SYCL=OFF \
  -DED_GGML_VULKAN=OFF \
  -DGGML_VULKAN=OFF \
  -DED_GGML_METAL=OFF \
  -DGGML_METAL=OFF \
  -DED_GGML_OPENCL=OFF \
  -DGGML_OPENCL=OFF \
  -DED_ENABLE_CUDNN_SDPA=OFF \
  -DED_ENABLE_CUDA_NORM=OFF \
  -DED_ENABLE_CUDA_ROPE=OFF \
  -DED_ENABLE_CUDA_MODULATION=OFF \
  -DED_ENABLE_CUDA_GRAPHS=OFF \
  -DED_ENABLE_CUDA_SAGE_ATTN=OFF \
  -DGGML_CUDA_GRAPHS=OFF \
  -DGGML_CUDA_NCCL=OFF \
  -DED_ENABLE_PARALLEL=OFF \
  -DED_ENABLE_NCCL=OFF \
  -DED_ENABLE_MPI=OFF \
  -DED_FETCH_CUDNN_FRONTEND=OFF \
  "-DCANN_INSTALL_DIR=${CANN_INSTALL_DIR}" \
  "-DSOC_TYPE=${SOC_TYPE}" \
  "-DUSE_ACL_GRAPH=${USE_ACL_GRAPH}" \
  "-DED_BENCHMARK_MARKERS=${ED_BENCHMARK_MARKERS}" \
  "$@"

"${CMAKE_BIN}" --build "${BUILD_DIR}" --parallel "${JOBS}"

"./${BUILD_DIR}/bin/ed-cli" --help
