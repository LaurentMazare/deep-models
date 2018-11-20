// PyTorch git version: 92dbd0219f6fbdb1db105386386ccf92c0758e86
//
// To build and run:
//   mkdir build
//   cd build
//   cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
//   make && ./infer

#include <chrono>
#include <stdio.h>
#include <torch/torch.h>
#include <torch/script.h>

void read_cifar_file(const char *path, torch::Tensor &images, torch::Tensor &labels) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) {
    printf("Error reading %s\n", path);
    exit(-1);
  }
  file.unsetf(std::ios::skipws);
  int64_t file_size = file.tellg();
  char *buffer = new char[file_size];
  file.seekg(0, std::ios::beg);
  file.read(buffer, file_size);
  file.close();
  torch::Tensor image = torch::empty({3, 32, 32}, torch::kByte);
  for (int i = 0; i < 10000; ++i) {
    memcpy(image.data_ptr(), buffer + 3073 * i + 1, 3072);
    images[i].copy_((image.toType(torch::kFloat) / 255.0 - 0.5) * 4.0);
    labels[i] = buffer[3073*i];
  }
  delete(buffer);
}

using namespace std::chrono;

int main(int argc, char *argv[]) {
  printf("Cuda: %d Cudnn: %d Devices: %d\n",
      torch::cuda::is_available(),
      torch::cuda::cudnn_is_available(),
      (int)torch::cuda::device_count());
  torch::Device device(torch::cuda::is_available() ? torch::kCUDA : torch::kCPU);
  at::globalContext().setBenchmarkCuDNN(true);
  torch::NoGradGuard no_grad;

  std::shared_ptr<torch::jit::script::Module> model = torch::jit::load("model.pt");
  model->to(device);

  // Warm-up.
  {
    torch::Tensor batch = torch::zeros({1, 3, 32, 32});
    auto batch_on_device = batch.to(device);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(batch_on_device);
    torch::Tensor t = model->forward(inputs).toTensor();
    int64_t pred = t.argmax(1).template item<int64_t>();
  }

  torch::Tensor cifar_images = torch::zeros({10000, 3, 32, 32}, torch::kFloat);
  torch::Tensor cifar_labels = torch::zeros({10000}, torch::kInt64);
  read_cifar_file("test_batch.bin", cifar_images, cifar_labels);
  auto start = high_resolution_clock::now();

  std::vector<int64_t> preds(10000, -1);
  for (int i = 0; i < 10000; ++i) {
    auto batch_on_device = cifar_images[i].view({1, 3, 32, 32}).to(device);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(batch_on_device);
    torch::Tensor t = model->forward(inputs).toTensor();
    preds[i] = t.argmax(1)[0].template item<int>();
  }
  auto stop = high_resolution_clock::now();
  auto duration = duration_cast<microseconds>(stop - start);
  int correct = 0;
  for (int i = 0; i < 10000; ++i) {
    int label = cifar_labels[i].template item<int64_t>();
    if (label == preds[i]) ++correct;
  }
  printf("accuracy: %.2f%% %luus per sample\n", 100 * correct / 10000.0, duration.count() / 10000);
}
