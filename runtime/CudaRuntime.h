#ifndef MLC_RUNTIME_CUDARUNTIME_H
#define MLC_RUNTIME_CUDARUNTIME_H

#include <cstdint>
#include <memory>
#include <string>

namespace mlc {

struct KernelBenchmarkConfig {
  unsigned warmupIterations = 25;
  unsigned timedIterations = 100;
};

struct KernelBenchmarkResult {
  double avgKernelMs = 0.0;
  float maxAbsError = 0.0f;
};

class CudaRuntime {
 public:
  CudaRuntime();
  ~CudaRuntime();

  CudaRuntime(const CudaRuntime &) = delete;
  CudaRuntime &operator=(const CudaRuntime &) = delete;

  bool initialize(std::string &error);
  bool loadModuleFromPtx(const std::string &ptxText, std::string &error);
  bool launchSoftmaxKernel(float sum, float &output, std::string &error);
  bool launchSoftmaxMemrefKernel(const float *inputHost, float *outputHost,
                                 std::int64_t n, float sum,
                                 std::string &error);
  bool benchmarkSoftmaxMemrefKernel(const float *inputHost, float *outputHost,
                                    std::int64_t n, float sum,
                                    const KernelBenchmarkConfig &config,
                                    KernelBenchmarkResult &result,
                                    std::string &error);

  static bool isCudaAvailable(std::string &reason);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mlc

#endif  // MLC_RUNTIME_CUDARUNTIME_H
