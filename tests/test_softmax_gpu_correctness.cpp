#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "Usage: test_softmax_gpu_correctness <mlc-demo> <input-mlir> <workdir>\n";
    return 1;
  }

  const std::string mlcDemo = argv[1];
  const std::string inputMlir = argv[2];
  const std::string workdir = argv[3];

  const std::string command =
      "\"" + mlcDemo + "\" --input \"" + inputMlir +
      "\" --output-dir \"" + workdir + "\" --mode optimized --sum 4.0 --verify";

  const int rc = std::system(command.c_str());
  if (rc != 0) {
    std::cerr << "mlc-demo command failed: " << command << "\n";
    return 1;
  }

  return 0;
}
