#include "runtime/CudaRuntime.h"

#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <string>

namespace mlc {
namespace {

using CUresult = int;
using CUdevice = int;
using CUcontext = void *;
using CUmodule = void *;
using CUfunction = void *;
using CUstream = void *;
using CUdeviceptr = std::uint64_t;

constexpr CUresult kCudaSuccess = 0;

template <typename Fn>
static bool loadSymbol(void *library,
                       const char *name,
                       Fn &fn,
                       std::string &error) {
  fn = reinterpret_cast<Fn>(dlsym(library, name));
  if (!fn) {
    error = std::string("Missing CUDA driver symbol: ") + name;
    return false;
  }
  return true;
}

}  // namespace

struct CudaRuntime::Impl {
  void *library = nullptr;

  CUcontext context = nullptr;
  CUmodule module = nullptr;

  CUresult (*cuInit)(unsigned int) = nullptr;
  CUresult (*cuDeviceGetCount)(int *) = nullptr;
  CUresult (*cuDeviceGet)(CUdevice *, int) = nullptr;
  CUresult (*cuCtxCreate)(CUcontext *, unsigned int, CUdevice) = nullptr;
  CUresult (*cuCtxDestroy)(CUcontext) = nullptr;
  CUresult (*cuModuleLoadDataEx)(CUmodule *, const void *, unsigned int, void *,
                                 void *) = nullptr;
  CUresult (*cuModuleUnload)(CUmodule) = nullptr;
  CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *) = nullptr;
  CUresult (*cuMemAlloc)(CUdeviceptr *, std::size_t) = nullptr;
  CUresult (*cuMemFree)(CUdeviceptr) = nullptr;
  CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void *, std::size_t) = nullptr;
  CUresult (*cuMemcpyDtoH)(void *, CUdeviceptr, std::size_t) = nullptr;
  CUresult (*cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int,
                             unsigned int, unsigned int, unsigned int,
                             unsigned int, CUstream, void **, void **) = nullptr;
  CUresult (*cuGetErrorString)(CUresult, const char **) = nullptr;

  std::string lastError(CUresult result) const {
    if (!cuGetErrorString) {
      return "CUDA error code " + std::to_string(result);
    }
    const char *msg = nullptr;
    if (cuGetErrorString(result, &msg) != kCudaSuccess || !msg) {
      return "CUDA error code " + std::to_string(result);
    }
    return std::string(msg);
  }

  bool loadApi(std::string &error) {
    const char *candidates[] = {
        "libcuda.so",
        "libcuda.so.1",
        "/usr/lib/x86_64-linux-gnu/libcuda.so.1",
        "/usr/lib64/libcuda.so.1",
        "/usr/local/cuda/lib64/stubs/libcuda.so",
    };

    for (const char *candidate : candidates) {
      library = dlopen(candidate, RTLD_NOW | RTLD_LOCAL);
      if (library) {
        break;
      }
    }

    if (!library) {
      error = "Could not load NVIDIA CUDA driver library (libcuda.so).";
      return false;
    }

    if (!loadSymbol(library, "cuInit", cuInit, error) ||
        !loadSymbol(library, "cuDeviceGetCount", cuDeviceGetCount, error) ||
        !loadSymbol(library, "cuDeviceGet", cuDeviceGet, error) ||
        !loadSymbol(library, "cuModuleLoadDataEx", cuModuleLoadDataEx, error) ||
        !loadSymbol(library, "cuModuleUnload", cuModuleUnload, error) ||
        !loadSymbol(library, "cuModuleGetFunction", cuModuleGetFunction, error) ||
        !loadSymbol(library, "cuLaunchKernel", cuLaunchKernel, error)) {
      return false;
    }

    cuGetErrorString =
        reinterpret_cast<CUresult (*)(CUresult, const char **)>(
            dlsym(library, "cuGetErrorString"));

    cuCtxCreate =
        reinterpret_cast<CUresult (*)(CUcontext *, unsigned int, CUdevice)>(
            dlsym(library, "cuCtxCreate_v2"));
    if (!cuCtxCreate) {
      cuCtxCreate =
          reinterpret_cast<CUresult (*)(CUcontext *, unsigned int, CUdevice)>(
              dlsym(library, "cuCtxCreate"));
    }

    cuCtxDestroy = reinterpret_cast<CUresult (*)(CUcontext)>(
        dlsym(library, "cuCtxDestroy_v2"));
    if (!cuCtxDestroy) {
      cuCtxDestroy =
          reinterpret_cast<CUresult (*)(CUcontext)>(dlsym(library, "cuCtxDestroy"));
    }

    cuMemAlloc = reinterpret_cast<CUresult (*)(CUdeviceptr *, std::size_t)>(
        dlsym(library, "cuMemAlloc_v2"));
    if (!cuMemAlloc) {
      cuMemAlloc = reinterpret_cast<CUresult (*)(CUdeviceptr *, std::size_t)>(
          dlsym(library, "cuMemAlloc"));
    }

    cuMemFree = reinterpret_cast<CUresult (*)(CUdeviceptr)>(
        dlsym(library, "cuMemFree_v2"));
    if (!cuMemFree) {
      cuMemFree =
          reinterpret_cast<CUresult (*)(CUdeviceptr)>(dlsym(library, "cuMemFree"));
    }

    cuMemcpyHtoD = reinterpret_cast<CUresult (*)(CUdeviceptr, const void *,
                                                 std::size_t)>(
        dlsym(library, "cuMemcpyHtoD_v2"));
    if (!cuMemcpyHtoD) {
      cuMemcpyHtoD = reinterpret_cast<CUresult (*)(CUdeviceptr, const void *,
                                                   std::size_t)>(
          dlsym(library, "cuMemcpyHtoD"));
    }

    cuMemcpyDtoH = reinterpret_cast<CUresult (*)(void *, CUdeviceptr,
                                                 std::size_t)>(
        dlsym(library, "cuMemcpyDtoH_v2"));
    if (!cuMemcpyDtoH) {
      cuMemcpyDtoH = reinterpret_cast<CUresult (*)(void *, CUdeviceptr,
                                                   std::size_t)>(
          dlsym(library, "cuMemcpyDtoH"));
    }

    if (!cuCtxCreate || !cuCtxDestroy || !cuMemAlloc || !cuMemFree ||
        !cuMemcpyHtoD || !cuMemcpyDtoH) {
      error = "Missing one or more required CUDA driver symbols for memory/context APIs.";
      return false;
    }

    return true;
  }

  void cleanup() {
    if (module && cuModuleUnload) {
      cuModuleUnload(module);
      module = nullptr;
    }

    if (context && cuCtxDestroy) {
      cuCtxDestroy(context);
      context = nullptr;
    }

    if (library) {
      dlclose(library);
      library = nullptr;
    }
  }
};

CudaRuntime::CudaRuntime() : impl_(std::make_unique<Impl>()) {}

CudaRuntime::~CudaRuntime() { impl_->cleanup(); }

bool CudaRuntime::initialize(std::string &error) {
  if (!impl_->library && !impl_->loadApi(error)) {
    return false;
  }

  CUresult rc = impl_->cuInit(0);
  if (rc != kCudaSuccess) {
    error = "cuInit failed: " + impl_->lastError(rc);
    return false;
  }

  int deviceCount = 0;
  rc = impl_->cuDeviceGetCount(&deviceCount);
  if (rc != kCudaSuccess) {
    error = "cuDeviceGetCount failed: " + impl_->lastError(rc);
    return false;
  }
  if (deviceCount <= 0) {
    error = "No CUDA devices available.";
    return false;
  }

  CUdevice device = 0;
  rc = impl_->cuDeviceGet(&device, 0);
  if (rc != kCudaSuccess) {
    error = "cuDeviceGet failed: " + impl_->lastError(rc);
    return false;
  }

  if (!impl_->context) {
    rc = impl_->cuCtxCreate(&impl_->context, 0, device);
    if (rc != kCudaSuccess) {
      error = "cuCtxCreate failed: " + impl_->lastError(rc);
      return false;
    }
  }

  return true;
}

bool CudaRuntime::loadModuleFromPtx(const std::string &ptxText,
                                    std::string &error) {
  if (ptxText.empty()) {
    error = "PTX text is empty.";
    return false;
  }

  if (!impl_->context) {
    error = "CUDA runtime is not initialized.";
    return false;
  }

  if (impl_->module) {
    impl_->cuModuleUnload(impl_->module);
    impl_->module = nullptr;
  }

  CUresult rc = impl_->cuModuleLoadDataEx(&impl_->module, ptxText.c_str(), 0,
                                          nullptr, nullptr);
  if (rc != kCudaSuccess) {
    error = "cuModuleLoadDataEx failed: " + impl_->lastError(rc);
    return false;
  }

  return true;
}

bool CudaRuntime::launchSoftmaxKernel(float sum,
                                      float &output,
                                      std::string &error) {
  if (!impl_->module) {
    error = "No CUDA module loaded.";
    return false;
  }

  CUfunction kernel = nullptr;
  CUresult rc = impl_->cuModuleGetFunction(&kernel, impl_->module, "softmax_kernel");
  if (rc != kCudaSuccess || !kernel) {
    error = "cuModuleGetFunction(softmax_kernel) failed: " + impl_->lastError(rc);
    return false;
  }

  CUdeviceptr dOutput = 0;
  rc = impl_->cuMemAlloc(&dOutput, sizeof(float));
  if (rc != kCudaSuccess) {
    error = "cuMemAlloc failed: " + impl_->lastError(rc);
    return false;
  }

  float zero = 0.0f;
  rc = impl_->cuMemcpyHtoD(dOutput, &zero, sizeof(float));
  if (rc != kCudaSuccess) {
    impl_->cuMemFree(dOutput);
    error = "cuMemcpyHtoD failed: " + impl_->lastError(rc);
    return false;
  }

  void *args[] = {&dOutput, &sum};
  rc = impl_->cuLaunchKernel(kernel,
                             1,
                             1,
                             1,
                             1,
                             1,
                             1,
                             0,
                             nullptr,
                             args,
                             nullptr);
  if (rc != kCudaSuccess) {
    impl_->cuMemFree(dOutput);
    error = "cuLaunchKernel failed: " + impl_->lastError(rc);
    return false;
  }

  rc = impl_->cuMemcpyDtoH(&output, dOutput, sizeof(float));
  impl_->cuMemFree(dOutput);
  if (rc != kCudaSuccess) {
    error = "cuMemcpyDtoH failed: " + impl_->lastError(rc);
    return false;
  }

  return true;
}

bool CudaRuntime::isCudaAvailable(std::string &reason) {
  CudaRuntime runtime;
  return runtime.initialize(reason);
}

}  // namespace mlc
