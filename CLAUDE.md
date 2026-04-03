# SpConv - Spatial Sparse Convolution Library

## Overview
SpConv (v2.3.8) implements high-performance sparse convolution operations for 1D/2D/3D/4D data, commonly used in 3D point cloud processing (e.g., autonomous driving). Depends on cumm for GEMM/convolution kernels. Uses PCCM for C++ code generation and pybind11 for bindings.

## Build System

### Build Command (pre-compiled wheel)
```bash
export SPCONV_DISABLE_JIT=1
export CUMM_CUDA_ARCH_LIST=all
export CUMM_CUDA_VERSION=12.8
export BOOST_ROOT=/path/to/boost_1_77_0
pip wheel . --no-deps -w dist/
```

### Key Environment Variables
| Variable | Purpose | Example |
|---|---|---|
| `CUMM_CUDA_VERSION` | Target CUDA version | `"12.8"`, `""` (CPU) |
| `SPCONV_DISABLE_JIT` | `"1"` for pre-compiled wheels | `"1"` |
| `CUMM_CUDA_ARCH_LIST` | GPU architectures | `"all"`, `"8.6"` |
| `BOOST_ROOT` | Boost 1.77.0 headers path | `/path/to/boost_1_77_0` |
| `SPCONV_PYTHON_LIST` | Python versions for build script | `"3.10;3.11;3.12;3.13"` |
| `SPCONV_VERSION_SUFFIX` | Dev version suffix | `"1.0"` → `2.3.8.dev1000` |

### Build Dependencies
- **Python**: pccm>=0.4.16, ccimport>=0.4.4, pybind11>=2.6.0, fire, numpy
- **Critical dep**: cumm>=0.8.2 (must be pre-installed for wheel builds)
- **C++**: NVIDIA CUDA Toolkit, C++ compiler
- **Headers**: Boost 1.77.0 (header-only, geometry module required)

### Docker Build (Linux manylinux wheels)
```bash
# Download Boost first
mkdir -p third_party
wget https://boostorg.jfrog.io/artifactory/main/release/1.77.0/source/boost_1_77_0.zip -O third_party/boost.zip
unzip third_party/boost.zip -d third_party/boost

docker run --rm \
  -e PLAT=manylinux_2_28_x86_64 \
  -e CUMM_CUDA_VERSION=12.8 \
  -e SPCONV_PYTHON_LIST="3.10;3.11;3.12;3.13" \
  -e BOOST_ROOT=/io/third_party/boost/boost_1_77_0 \
  -v $(pwd):/io \
  scrin/manylinux2014-cuda:cu128-devel-1.0.0 \
  bash -c "source /etc/bashrc && /io/tools/build-wheels.sh"
```

### Build Order
cumm must be built and installed BEFORE building spconv (spconv imports from cumm at build time).

### Platform Tags
- CUDA < 12.4: `manylinux2014_x86_64`
- CUDA ≥ 12.4: `manylinux_2_28_x86_64`

## Project Structure
- `spconv/` - Python package
  - `csrc/` - C++ source definitions via PCCM
    - `sparse/all.py` (99KB) - SpconvOps main binding
    - `sparse/convops.py` (99KB) - GemmTuner, ConvTuner, ConvGemmOps
    - `sparse/indices.py` (84KB) - Sparse index pair generation
    - `sparse/alloc.py` - Memory allocation (thrust, external)
  - `pytorch/` - PyTorch integration, quantization
  - `core.py` - Kernel parameter definitions (SIMT, Volta, Turing, Ampere)
  - `constants.py` - Package constants, Boost path, JIT settings
  - `core_cc/` - Generated C++ extensions (build output)
- `test/` - Test suite
- `example/` - MNIST, sparse conv examples
- `tools/` - Build scripts
- `docs/` - API, performance, quantization guides

## Key Files
- `setup.py` - Package build, cumm dependency constraint, kernel compilation
- `spconv/core.py` - **Critical**: GEMM/conv kernel parameters for all GPU architectures
- `spconv/constants.py` - Boost path, JIT settings, weight layout
- `tools/build-wheels.sh` - Linux wheel build script (uses SPCONV_PYTHON_LIST)

## Kernel Architecture Support (core.py)
- `SHUFFLE_SIMT_PARAMS` - f32/f16 kernels for all GPUs (fallback)
- `SHUFFLE_VOLTA_PARAMS` - Volta tensor core (sm_70)
- `SHUFFLE_TURING_PARAMS` - Turing tensor core (sm_75)
- `SHUFFLE_AMPERE_PARAMS` - Ampere (currently empty, uses NVRTC)
- `IMPLGEMM_*_PARAMS` - Implicit GEMM variants for each arch

## Package Naming
- CPU: `spconv`
- CUDA: `spconv-cu{VER}` (e.g., `spconv-cu128`)
