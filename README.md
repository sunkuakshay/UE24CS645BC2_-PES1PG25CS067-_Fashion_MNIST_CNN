# CNN from Scratch — Fashion MNIST

**Subject:** Deep Learning Theory and Practice (DLTP) — Assignment 1  
**Dataset:** [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist)

---

## Project Overview

This project implements a **Convolutional Neural Network (CNN) entirely from scratch using NumPy** — no PyTorch, no TensorFlow. Every layer, every forward pass, and every gradient computation is hand-coded to demonstrate a deep understanding of how CNNs work under the hood.

The network is trained and evaluated on the **Fashion MNIST** dataset, which consists of 70,000 grayscale 28×28 images across 10 clothing/accessory classes.

---

## Architecture

```
Input (N, 1, 28, 28)
    │
    ▼
ConvLayer(in=1, filters=8, kernel=3×3, pad=1)
    │  ReLU activation
    ▼
MaxPoolLayer(pool=2×2, stride=2)           → (N, 8, 14, 14)
    │
    ▼
ConvLayer(in=8, filters=16, kernel=3×3, pad=1)
    │  ReLU activation
    ▼
MaxPoolLayer(pool=2×2, stride=2)           → (N, 16, 7, 7)
    │
    ▼
FlattenLayer                               → (N, 784)
    │
    ▼
FCLayer(784 → 128, ReLU)
    │
    ▼
FCLayer(128 → 10, Linear)                 → Logits
    │
    ▼
Softmax + Cross-Entropy Loss
```

---

## Implemented Components

| Component | File Section | Description |
|-----------|-------------|-------------|
| **Convolution Layer** | `ConvLayer` | Forward (im2col matmul) + Backward (gradient via col2im) |
| **Max Pooling** | `MaxPoolLayer` | Forward (argmax) + Backward (gradient mask routing) |
| **Flatten** | `FlattenLayer` | Reshape + inverse reshape for backprop |
| **Fully Connected** | `FCLayer` | Linear + optional ReLU, with dW/db gradients |
| **Softmax + Loss** | `cross_entropy_loss` | Numerically stable softmax cross-entropy |
| **Training Loop** | `train()` | Mini-batch SGD with shuffled data |
| **Evaluation** | `evaluate()` | Overall + per-class accuracy |

---

## Key Concepts Demonstrated

### Convolution Operation
Convolution is a linear operation where a small filter (kernel) slides over the input image, computing dot products at each position. This allows the network to detect local spatial patterns (edges, textures) regardless of their position.

The formula for a 2D convolution at position (i, j) for filter f is:
```
out[f, i, j] = Σ_c Σ_m Σ_n  W[f, c, m, n] · x[c, i·s+m, j·s+n]  +  b[f]
```
where `s` is the stride and the sum is over channels `c` and kernel positions `m, n`.

### im2col Optimisation
Rather than using nested Python loops for every kernel position, we use the **im2col** trick: rearrange image patches into a 2D matrix so the entire convolution becomes a single matrix multiply. This dramatically improves performance.

### Backpropagation in CNN
- **Conv backward:** gradient w.r.t. weights is a correlation between input patches and output gradients; gradient w.r.t. input uses col2im to scatter gradients back.
- **MaxPool backward:** gradients flow only through positions that held the maximum value (all others receive zero gradient).
- **FC backward:** standard matrix transpose operations.

### Activation Function
**ReLU** (Rectified Linear Unit): `f(x) = max(0, x)`  
Chosen for hidden layers because it is computationally cheap, does not suffer from vanishing gradients, and works well with He weight initialisation.

---

## Requirements

```
Python  >= 3.8
numpy   >= 1.21
```

No other dependencies. The script auto-downloads Fashion MNIST.

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/<YOUR_USERNAME>/UE24CS645BC2_<USN>_Fashion_MNIST_CNN.git
cd UE24CS645BC2_<USN>_Fashion_MNIST_CNN
```

### 2. Install dependencies
```bash
pip install numpy
```

### 3. Run training and evaluation
```bash
python cnn_fashion_mnist.py
```

The script will:
1. Automatically download Fashion MNIST into a `data/` folder
2. Build the CNN from scratch
3. Train for 5 epochs using mini-batch SGD
4. Print per-epoch loss
5. Evaluate on the test set and print per-class accuracy

### 4. (Optional) Full training
By default, a subset of 10,000 training samples is used for speed. To train on the full 60,000 samples, open `cnn_fashion_mnist.py` and set:
```python
TRAIN_SUBSET = None   # or 60000
```

---

## Expected Output (demo subset, 5 epochs)

```
============================================================
  CNN from Scratch — Fashion MNIST
============================================================

[1] Loading Fashion MNIST …
    Train : (10000, 1, 28, 28)  Labels: (10000,)
    Test  : (10000, 1, 28, 28)   Labels: (10000,)

[2] Building CNN …
    Architecture:
      Conv(1→8,  3×3, pad=1) → ReLU → MaxPool(2×2)
      Conv(8→16, 3×3, pad=1) → ReLU → MaxPool(2×2)
      Flatten → FC(784→128, ReLU) → FC(128→10)

[3] Training …
  Epoch 1 | Batch  100/156 | Loss: 1.8342
  ...
Epoch 5/5 — Avg Loss: 0.6123

[4] Evaluating on Test Set …
Overall Accuracy: ~72–78%
```

> Accuracy on the full dataset (60,000 training samples) is typically **75–82%** after 5–10 epochs with this architecture.

---

## Fashion MNIST Classes

| Label | Class |
|-------|-------|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## Submission Details

- **Name:** [Your Name]  
- **USN:** [Your USN]  
- **Email to:** paragjainpes@gmail.com  
- **Subject:** DLTP_Assignment_1  
- **Deadline:** Friday, 15th May 2026, 11:59 PM
