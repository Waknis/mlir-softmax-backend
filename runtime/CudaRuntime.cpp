#include "runtime/CudaRuntime.h"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <array>
#include <chrono>
#include <string>
#include <vector>

namespace mlc {
namespace {

using CUresult = int;
using CUdevice = int;
using CUcontext = void *;
using CUmodule = void *;
using CUfunction = void *;
using CUstream = void *;
using CUevent = void *;
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
  CUresult (*cuCtxSynchronize)() = nullptr;
  CUresult (*cuModuleLoadDataEx)(CUmodule *, const void *, unsigned int, void *,
                                 void *) = nullptr;
  CUresult (*cuModuleUnload)(CUmodule) = nullptr;
  CUresult (*cuModuleGetFunction)(CUfunction *, CUmodule, const char *) = nullptr;
  CUresult (*cuEventCreate)(CUevent *, unsigned int) = nullptr;
  CUresult (*cuEventDestroy)(CUevent) = nullptr;
  CUresult (*cuEventRecord)(CUevent, CUstream) = nullptr;
  CUresult (*cuEventSynchronize)(CUevent) = nullptr;
  CUresult (*cuEventElapsedTime)(float *, CUevent, CUevent) = nullptr;
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
    cuCtxSynchronize =
        reinterpret_cast<CUresult (*)()>(dlsym(library, "cuCtxSynchronize"));

    cuEventCreate =
        reinterpret_cast<CUresult (*)(CUevent *, unsigned int)>(
            dlsym(library, "cuEventCreate"));
    cuEventDestroy =
        reinterpret_cast<CUresult (*)(CUevent)>(dlsym(library, "cuEventDestroy_v2"));
    if (!cuEventDestroy) {
      cuEventDestroy =
          reinterpret_cast<CUresult (*)(CUevent)>(dlsym(library, "cuEventDestroy"));
    }
    cuEventRecord =
        reinterpret_cast<CUresult (*)(CUevent, CUstream)>(
            dlsym(library, "cuEventRecord"));
    cuEventSynchronize =
        reinterpret_cast<CUresult (*)(CUevent)>(
            dlsym(library, "cuEventSynchronize"));
    cuEventElapsedTime =
        reinterpret_cast<CUresult (*)(float *, CUevent, CUevent)>(
            dlsym(library, "cuEventElapsedTime"));

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

    if (!cuCtxCreate || !cuCtxDestroy || !cuCtxSynchronize || !cuMemAlloc || !cuMemFree ||
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

  enum CuJitOption {
    kCuJitInfoLogBuffer = 3,
    kCuJitInfoLogBufferSizeBytes = 4,
    kCuJitErrorLogBuffer = 5,
    kCuJitErrorLogBufferSizeBytes = 6,
  };

  std::array<char, 8192> infoLog{};
  std::array<char, 8192> errorLog{};
  auto infoLogSizeValue = static_cast<std::uintptr_t>(infoLog.size());
  auto errorLogSizeValue = static_cast<std::uintptr_t>(errorLog.size());

  std::array<int, 4> options = {
      kCuJitErrorLogBuffer,
      kCuJitErrorLogBufferSizeBytes,
      kCuJitInfoLogBuffer,
      kCuJitInfoLogBufferSizeBytes,
  };
  // CUDA driver API expects size options as integer values cast to void*,
  // not pointers to host-side variables.
  std::array<void *, 4> optionValues = {
      errorLog.data(),
      reinterpret_cast<void *>(errorLogSizeValue),
      infoLog.data(),
      reinterpret_cast<void *>(infoLogSizeValue),
  };

  CUresult rc = impl_->cuModuleLoadDataEx(&impl_->module,
                                          ptxText.c_str(),
                                          static_cast<unsigned int>(options.size()),
                                          options.data(),
                                          optionValues.data());
  if (rc != kCudaSuccess) {
    error = "cuModuleLoadDataEx failed: " + impl_->lastError(rc);
    if (!errorLog.empty() && errorLog.front() != '\0') {
      error += "\nPTX JIT error log:\n";
      error += errorLog.data();
    }
    if (!infoLog.empty() && infoLog.front() != '\0') {
      error += "\nPTX JIT info log:\n";
      error += infoLog.data();
    }
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

bool CudaRuntime::launchSoftmaxMemrefKernel(const float *inputHost,
                                            float *outputHost,
                                            std::int64_t n, float sum,
                                            std::string &error) {
  if (!impl_->module) {
    error = "No CUDA module loaded.";
    return false;
  }

  CUfunction kernel = nullptr;
  CUresult rc =
      impl_->cuModuleGetFunction(&kernel, impl_->module, "softmax_kernel");
  if (rc != kCudaSuccess || !kernel) {
    error = "cuModuleGetFunction(softmax_kernel) failed: " +
            impl_->lastError(rc);
    return false;
  }

  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);

  CUdeviceptr dInput = 0;
  rc = impl_->cuMemAlloc(&dInput, bytes);
  if (rc != kCudaSuccess) {
    error = "cuMemAlloc(input) failed: " + impl_->lastError(rc);
    return false;
  }

  CUdeviceptr dOutput = 0;
  rc = impl_->cuMemAlloc(&dOutput, bytes);
  if (rc != kCudaSuccess) {
    impl_->cuMemFree(dInput);
    error = "cuMemAlloc(output) failed: " + impl_->lastError(rc);
    return false;
  }

  rc = impl_->cuMemcpyHtoD(dInput, inputHost, bytes);
  if (rc != kCudaSuccess) {
    impl_->cuMemFree(dInput);
    impl_->cuMemFree(dOutput);
    error = "cuMemcpyHtoD(input) failed: " + impl_->lastError(rc);
    return false;
  }

  // Zero output buffer.
  std::vector<float> zeros(static_cast<std::size_t>(n), 0.0f);
  rc = impl_->cuMemcpyHtoD(dOutput, zeros.data(), bytes);
  if (rc != kCudaSuccess) {
    impl_->cuMemFree(dInput);
    impl_->cuMemFree(dOutput);
    error = "cuMemcpyHtoD(output) failed: " + impl_->lastError(rc);
    return false;
  }

  // Kernel args: (ptr input, ptr output, i64 n, f32 sum)
  std::int64_t nArg = n;
  void *args[] = {&dInput, &dOutput, &nArg, &sum};

  // Launch with enough threads to cover n elements.
  // The kernel itself is a sequential loop, so grid=1, block=1 is correct
  // for the current scf.for-based IR (no GPU parallelism yet).
  rc = impl_->cuLaunchKernel(kernel,
                             1, 1, 1,   // grid
                             1, 1, 1,   // block
                             0, nullptr, args, nullptr);
  if (rc != kCudaSuccess) {
    impl_->cuMemFree(dInput);
    impl_->cuMemFree(dOutput);
    error = "cuLaunchKernel failed: " + impl_->lastError(rc);
    return false;
  }

  rc = impl_->cuMemcpyDtoH(outputHost, dOutput, bytes);
  impl_->cuMemFree(dInput);
  impl_->cuMemFree(dOutput);
  if (rc != kCudaSuccess) {
    error = "cuMemcpyDtoH(output) failed: " + impl_->lastError(rc);
    return false;
  }

  return true;
}

bool CudaRuntime::benchmarkSoftmaxMemrefKernel(
    const float *inputHost,
    float *outputHost,
    std::int64_t n,
    float sum,
    const KernelBenchmarkConfig &config,
    KernelBenchmarkResult &result,
    std::string &error) {
  if (!inputHost || !outputHost) {
    error = "Host input/output buffers must be non-null.";
    return false;
  }
  if (n <= 0) {
    error = "Benchmark length must be positive.";
    return false;
  }
  if (config.timedIterations == 0) {
    error = "Benchmark timedIterations must be greater than zero.";
    return false;
  }
  if (!impl_->module) {
    error = "No CUDA module loaded.";
    return false;
  }

  CUfunction kernel = nullptr;
  CUresult rc =
      impl_->cuModuleGetFunction(&kernel, impl_->module, "softmax_kernel");
  if (rc != kCudaSuccess || !kernel) {
    error = "cuModuleGetFunction(softmax_kernel) failed: " +
            impl_->lastError(rc);
    return false;
  }

  const std::size_t bytes = static_cast<std::size_t>(n) * sizeof(float);
  CUdeviceptr dInput = 0;
  CUdeviceptr dOutput = 0;
  CUevent startEvent = nullptr;
  CUevent stopEvent = nullptr;

  auto cleanup = [&]() {
    if (startEvent && impl_->cuEventDestroy) {
      impl_->cuEventDestroy(startEvent);
      startEvent = nullptr;
    }
    if (stopEvent && impl_->cuEventDestroy) {
      impl_->cuEventDestroy(stopEvent);
      stopEvent = nullptr;
    }
    if (dInput) {
      impl_->cuMemFree(dInput);
      dInput = 0;
    }
    if (dOutput) {
      impl_->cuMemFree(dOutput);
      dOutput = 0;
    }
  };

  rc = impl_->cuMemAlloc(&dInput, bytes);
  if (rc != kCudaSuccess) {
    error = "cuMemAlloc(input) failed: " + impl_->lastError(rc);
    cleanup();
    return false;
  }

  rc = impl_->cuMemAlloc(&dOutput, bytes);
  if (rc != kCudaSuccess) {
    error = "cuMemAlloc(output) failed: " + impl_->lastError(rc);
    cleanup();
    return false;
  }

  rc = impl_->cuMemcpyHtoD(dInput, inputHost, bytes);
  if (rc != kCudaSuccess) {
    error = "cuMemcpyHtoD(input) failed: " + impl_->lastError(rc);
    cleanup();
    return false;
  }

  bool useEvents = impl_->cuEventCreate && impl_->cuEventDestroy &&
                   impl_->cuEventRecord && impl_->cuEventSynchronize &&
                   impl_->cuEventElapsedTime;
  if (useEvents) {
    rc = impl_->cuEventCreate(&startEvent, 0);
    if (rc != kCudaSuccess) {
      error = "cuEventCreate(start) failed: " + impl_->lastError(rc);
      cleanup();
      return false;
    }
    rc = impl_->cuEventCreate(&stopEvent, 0);
    if (rc != kCudaSuccess) {
      error = "cuEventCreate(stop) failed: " + impl_->lastError(rc);
      cleanup();
      return false;
    }
  }

  std::int64_t nArg = n;
  void *args[] = {&dInput, &dOutput, &nArg, &sum};

  for (unsigned iter = 0; iter < config.warmupIterations; ++iter) {
    rc = impl_->cuLaunchKernel(kernel,
                               1, 1, 1,
                               1, 1, 1,
                               0, nullptr, args, nullptr);
    if (rc != kCudaSuccess) {
      error = "cuLaunchKernel warmup failed: " + impl_->lastError(rc);
      cleanup();
      return false;
    }
  }
  rc = impl_->cuCtxSynchronize();
  if (rc != kCudaSuccess) {
    error = "cuCtxSynchronize after warmup failed: " + impl_->lastError(rc);
    cleanup();
    return false;
  }

  double totalKernelMs = 0.0;
  for (unsigned iter = 0; iter < config.timedIterations; ++iter) {
    auto hostStart = std::chrono::high_resolution_clock::now();
    if (useEvents) {
      rc = impl_->cuEventRecord(startEvent, nullptr);
      if (rc != kCudaSuccess) {
        error = "cuEventRecord(start) failed: " + impl_->lastError(rc);
        cleanup();
        return false;
      }
    }

    rc = impl_->cuLaunchKernel(kernel,
                               1, 1, 1,
                               1, 1, 1,
                               0, nullptr, args, nullptr);
    if (rc != kCudaSuccess) {
      error = "cuLaunchKernel failed: " + impl_->lastError(rc);
      cleanup();
      return false;
    }

    if (useEvents) {
      rc = impl_->cuEventRecord(stopEvent, nullptr);
      if (rc != kCudaSuccess) {
        error = "cuEventRecord(stop) failed: " + impl_->lastError(rc);
        cleanup();
        return false;
      }
      rc = impl_->cuEventSynchronize(stopEvent);
      if (rc != kCudaSuccess) {
        error = "cuEventSynchronize(stop) failed: " + impl_->lastError(rc);
        cleanup();
        return false;
      }

      float kernelMs = 0.0f;
      rc = impl_->cuEventElapsedTime(&kernelMs, startEvent, stopEvent);
      if (rc != kCudaSuccess) {
        error = "cuEventElapsedTime failed: " + impl_->lastError(rc);
        cleanup();
        return false;
      }
      totalKernelMs += static_cast<double>(kernelMs);
    } else {
      rc = impl_->cuCtxSynchronize();
      if (rc != kCudaSuccess) {
        error = "cuCtxSynchronize failed: " + impl_->lastError(rc);
        cleanup();
        return false;
      }
      auto hostEnd = std::chrono::high_resolution_clock::now();
      totalKernelMs +=
          std::chrono::duration<double, std::milli>(hostEnd - hostStart).count();
    }
  }

  rc = impl_->cuMemcpyDtoH(outputHost, dOutput, bytes);
  if (rc != kCudaSuccess) {
    error = "cuMemcpyDtoH(output) failed: " + impl_->lastError(rc);
    cleanup();
    return false;
  }

  cleanup();

  float maxAbsError = 0.0f;
  for (std::int64_t i = 0; i < n; ++i) {
    const std::size_t index = static_cast<std::size_t>(i);
    const float expected = inputHost[index] / sum;
    const float absErr = std::fabs(outputHost[index] - expected);
    if (absErr > maxAbsError) {
      maxAbsError = absErr;
    }
  }

  result.avgKernelMs =
      totalKernelMs / static_cast<double>(config.timedIterations);
  result.maxAbsError = maxAbsError;
  return true;
}

bool CudaRuntime::isCudaAvailable(std::string &reason) {
  CudaRuntime runtime;
  return runtime.initialize(reason);
}

}  // namespace mlc
