# cifar10-fast-jit

This project demonstrates how to train a ResNet model on
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) using PyTorch Python
api, save the trained model via Torch Script and run it in C++ on the test part
of the CIFAR-10 dataset.

The resulting model takes 828 microseconds to run per image on a single
P100 GPU (this includes the time to copy the image from the main memory to GPU
memory). This timing has been averaged over the 10000 images contained on the
test dataset of CIFAR-10.  Each image is processed independently and the next
image only starts being processed after that the first image has been handled.

The ResNet model is a trimmed down version of ResNet-9 from the
[cifar10-fast repo](https://github.com/davidcpage/cifar10-fast).

The main differences are that the model is smaller and is trained for longer
to reach more than 94% accuracy on the test set of CIFAR-10.
On 5 training runs the model reached accuracies from 94.12% to 94.72%.

## Run Inference

To reproduce inference timings download the PyTorch C++ api from the
[PyTorch website](https://pytorch.org/get-started/locally/), extract it
in a `libtorch` directory and run the following steps.

```bash
# Download a pre-trained model or re-train it with 'python train.py'
wget https://github.com/LaurentMazare/deep-models/releases/download/v0.0.1/model.pt
# Download the CIFAR-10 test dataset.
wget https://github.com/LaurentMazare/deep-models/releases/download/v0.0.1/test_batch.bin

mkdir build
cd build
cmake -DCMAKE_PREFIX_PATH=/absolute/path/to/libtorch ..
make
cd ..
./build/infer
```

The returned accuracy should be ~94.53%.
On a single P100 the running time is 828 microseconds per image.
On 10 runs the timings were no more than a 2 microseconds per image apart.

This was run on [PyTorch](https://github.com/pytorch/pytorch)
commit id 92dbd0219f6fbdb1db105386386ccf92c0758e86.
The test set from CIFAR-10 can also be downloaded from the whole
[CIFAR-10 dataset](https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz).
## Run Training

The `model.pt` model file can be produced by running the Python training script via:

```bash
python train.py
```

Note that this requires the 1.0 preview version of PyTorch.

## Low Hanging Fruits

It should be easy to improve the timings above, for example by:

- Running on a V100 rather than a P100.
- Using half-precision floats rather than single-precision. *Update
  2018-11-30:* this does not seem to improve the results on a P100. Maybe the
  GPU is not used enough for it to make a difference.
