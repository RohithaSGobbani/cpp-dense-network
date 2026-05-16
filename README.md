# cpp-dense-network
#  Bare-Metal C++ Neural Network Inference Engine

A custom-built, bare-metal fully connected neural network (Dense) inference engine written entirely from scratch in C++. **No TensorFlow, no PyTorch, no NumPy, and no external BLAS libraries.** Just raw arrays, memory management, and custom matrix math.

This backend is paired with a modern Python/Streamlit web UI that captures real-time handwriting, processes it using historical computer vision algorithms, and pipes it directly into the C++ executable.

##  Performance
* **Topology:** 2-Layer Fully Connected Network (Dense -> ReLU -> Dense)
* **Dataset:** MNIST
* **Benchmark Accuracy:** **95.9%** on the 10,000-image unseen test set (hitting the mathematical ceiling for non-convolutional spatial topologies).
* **Speed:** Blazing fast bare-metal execution compiled with `-O3` optimization.

##  Architecture & Features
* **Custom Matrix Math Library:** Built-in dot-products, bias addition, and ReLU activation functions utilizing raw C++ 1D arrays for CPU cache optimization.
* **Byte-Level Binary Parsing:** Custom file loaders that bypass standard libraries to directly parse raw MNIST `idx` binary files, including manual Big-Endian to Little-Endian 32-bit hardware swapping.
* **The "Yann LeCun" Preprocessing Pipeline:** The Python frontend perfectly replicates the original 1990s MNIST digit standardization algorithm:
  1. Ink bounding-box extraction.
  2. Aspect-ratio-preserving downscale to a 20x20 bounding limit.
  3. Center-of-mass alignment into a 28x28 mathematical canvas.
  4. Gaussian blurring to simulate 1990s scanner ink bleed.
* **Subprocess Pipelining:** Python standardizes the data into a normalized `0.0` to `1.0` float array and pipes it to the C++ runtime environment for instant inference.

##  Project Structure
```text
├── app.py                      # Streamlit frontend & preprocessing pipeline
├── matrix.cpp                  # Core C++ inference engine & matrix math
├── custom_digit.txt            # I/O bridge for Python -> C++ communication
├── weights_archive/            # Pre-trained W1, b1, W2, b2 weights
└── dataset/                    # MNIST idx binary files (ignored in git)
