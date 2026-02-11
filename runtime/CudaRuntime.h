#ifndef MLC_RUNTIME_CUDARUNTIME_H
#define MLC_RUNTIME_CUDARUNTIME_H

#include <memory>
#include <string>

namespace mlc {

class CudaRuntime {
 public:
  CudaRuntime();
  ~CudaRuntime();

  CudaRuntime(const CudaRuntime &) = delete;
  CudaRuntime &operator=(const CudaRuntime &) = delete;

  bool initialize(std::string &error);
  bool loadModuleFromPtx(const std::string &ptxText, std::string &error);
  bool launchSoftmaxKernel(float sum, float &output, std::string &error);

  static bool isCudaAvailable(std::string &reason);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace mlc

#endif  // MLC_RUNTIME_CUDARUNTIME_H
