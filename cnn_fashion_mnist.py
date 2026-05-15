"""
CNN from Scratch for Fashion MNIST
===================================
Implements Convolution, Pooling, Flatten, Fully Connected layers
with forward and backward passes (backpropagation), trained on Fashion MNIST.

Author  : [Sunku Peda Akshay]
USN     : [PES1PG25CS067]
Subject : DLTP Assignment 1
"""

import numpy as np
import urllib.request
import gzip
import os
import struct

# ─────────────────────────────────────────────────────────────────────────────
# 1.  DATA LOADING
# ─────────────────────────────────────────────────────────────────────────────

FASHION_MNIST_URLS = {
    "train_images": "http://fashion-mnist.s3-website.eu-west-1.amazonaws.com/train-images-idx3-ubyte.gz",
    "train_labels": "http://fashion-mnist.s3-website.eu-west-1.amazonaws.com/train-labels-idx1-ubyte.gz",
    "test_images":  "http://fashion-mnist.s3-website.eu-west-1.amazonaws.com/t10k-images-idx3-ubyte.gz",
    "test_labels":  "http://fashion-mnist.s3-website.eu-west-1.amazonaws.com/t10k-labels-idx1-ubyte.gz",
}

CLASS_NAMES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal",      "Shirt",   "Sneaker",  "Bag",   "Ankle boot"
]


def _download(name, url, data_dir="data"):
    os.makedirs(data_dir, exist_ok=True)
    gz_path = os.path.join(data_dir, name + ".gz")
    if not os.path.exists(gz_path):
        print(f"  Downloading {name} …")
        urllib.request.urlretrieve(url, gz_path)
    return gz_path


def load_fashion_mnist(data_dir="data"):
    """Download (if needed) and load Fashion MNIST. Returns numpy arrays."""
    paths = {k: _download(k, v, data_dir) for k, v in FASHION_MNIST_URLS.items()}

    def read_images(path):
        with gzip.open(path, "rb") as f:
            magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(n, 1, rows, cols)

    def read_labels(path):
        with gzip.open(path, "rb") as f:
            magic, n = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)

    X_train = read_images(paths["train_images"]).astype(np.float32) / 255.0
    y_train = read_labels(paths["train_labels"])
    X_test  = read_images(paths["test_images"]).astype(np.float32) / 255.0
    y_test  = read_labels(paths["test_labels"])
    return X_train, y_train, X_test, y_test


# ─────────────────────────────────────────────────────────────────────────────
# 2.  ACTIVATION FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def relu(x):
    return np.maximum(0, x)


def relu_backward(dout, x):
    return dout * (x > 0)


def softmax(x):
    # Numerically stable softmax
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


# ─────────────────────────────────────────────────────────────────────────────
# 3.  CONVOLUTION LAYER
# ─────────────────────────────────────────────────────────────────────────────

class ConvLayer:
    """
    2-D Convolutional Layer.

    Parameters
    ----------
    in_channels  : number of input channels
    num_filters  : number of convolutional filters (output channels)
    kernel_size  : square kernel size (k × k)
    stride       : convolution stride (default 1)
    padding      : zero-padding added to each side (default 0)
    """

    def __init__(self, in_channels, num_filters, kernel_size, stride=1, padding=0):
        self.in_channels = in_channels
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # He initialisation — good for ReLU
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.W = np.random.randn(num_filters, in_channels, kernel_size, kernel_size).astype(np.float32) * scale
        self.b = np.zeros((num_filters, 1), dtype=np.float32)

        # Gradients
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

        # Cache for backward pass
        self._cache = {}

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _pad(x, pad):
        if pad == 0:
            return x
        return np.pad(x, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode="constant")

    def _out_size(self, h, w):
        oh = (h + 2 * self.padding - self.kernel_size) // self.stride + 1
        ow = (w + 2 * self.padding - self.kernel_size) // self.stride + 1
        return oh, ow

    # ── im2col / col2im ──────────────────────────────────────────────────────

    def _im2col(self, x_padded, oh, ow):
        """
        Rearrange image patches into columns so the convolution becomes
        a single matrix multiply.  Returned shape: (N, C·k·k, oh·ow).
        """
        N, C, H_pad, W_pad = x_padded.shape
        k = self.kernel_size
        s = self.stride
        col = np.zeros((N, C, k, k, oh, ow), dtype=np.float32)
        for i in range(k):
            i_max = i + s * oh
            for j in range(k):
                j_max = j + s * ow
                col[:, :, i, j, :, :] = x_padded[:, :, i:i_max:s, j:j_max:s]
        return col.reshape(N, C * k * k, oh * ow)

    def _col2im(self, dcol, x_padded_shape):
        """Inverse of im2col — accumulate gradients back into image space."""
        N, C, H_pad, W_pad = x_padded_shape
        k = self.kernel_size
        s = self.stride
        oh = (H_pad - k) // s + 1
        ow = (W_pad - k) // s + 1
        dcol_r = dcol.reshape(N, C, k, k, oh, ow)
        dx_padded = np.zeros((N, C, H_pad, W_pad), dtype=np.float32)
        for i in range(k):
            i_max = i + s * oh
            for j in range(k):
                j_max = j + s * ow
                dx_padded[:, :, i:i_max:s, j:j_max:s] += dcol_r[:, :, i, j, :, :]
        return dx_padded

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x):
        """
        Forward pass of the convolution.

        x : (N, C, H, W)
        Returns output : (N, F, OH, OW)
        """
        N, C, H, W = x.shape
        oh, ow = self._out_size(H, W)
        x_padded = self._pad(x, self.padding)

        col = self._im2col(x_padded, oh, ow)          # (N, C·k·k, oh·ow)
        W_col = self.W.reshape(self.num_filters, -1)   # (F, C·k·k)

        # out[n, f, i, j] = W_col[f] · col[n, :, i*ow+j]  + b[f]
        out = W_col @ col                               # (N, F, oh·ow)  via broadcast
        # Actually shapes: W_col (F, Ck²), col (N, Ck², oh·ow)
        # We need (N, F, oh·ow):
        out = np.einsum("fc,ncp->nfp", W_col, col)    # (N, F, oh·ow)
        out = out.reshape(N, self.num_filters, oh, ow)
        out += self.b.reshape(1, self.num_filters, 1, 1)

        self._cache = {"x": x, "x_padded": x_padded, "col": col}
        return out

    # ── backward ─────────────────────────────────────────────────────────────

    def backward(self, dout):
        """
        Backward pass (backpropagation through convolution).

        dout : (N, F, OH, OW)
        Returns dx : (N, C, H, W)
        """
        x       = self._cache["x"]
        x_padded= self._cache["x_padded"]
        col     = self._cache["col"]

        N, F, oh, ow = dout.shape
        dout_r = dout.reshape(N, F, oh * ow)           # (N, F, oh·ow)

        # Gradient w.r.t. bias
        self.db = dout_r.sum(axis=(0, 2)).reshape(F, 1)

        # Gradient w.r.t. weights
        # dW[f] = sum_n  dout_r[n,f,:] · col[n,:,:]ᵀ   → (F, C·k²)
        self.dW = np.einsum("nfp,ncp->fc", dout_r, col).reshape(self.W.shape)

        # Gradient w.r.t. input (col space)
        W_col = self.W.reshape(F, -1)                   # (F, C·k²)
        dcol  = np.einsum("fc,nfp->ncp", W_col, dout_r) # (N, C·k², oh·ow)

        # Convert back from col to image
        dx_padded = self._col2im(dcol, x_padded.shape)

        # Remove padding
        p = self.padding
        if p == 0:
            dx = dx_padded
        else:
            dx = dx_padded[:, :, p:-p, p:-p]
        return dx


# ─────────────────────────────────────────────────────────────────────────────
# 4.  MAX POOLING LAYER
# ─────────────────────────────────────────────────────────────────────────────

class MaxPoolLayer:
    """
    2-D Max Pooling Layer.

    Parameters
    ----------
    pool_size : square pooling window (default 2)
    stride    : stride of the pooling window (default 2)
    """

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride = stride
        self._cache = {}

    def forward(self, x):
        """
        Forward pass.

        x      : (N, C, H, W)
        Returns: (N, C, OH, OW)  where OH = (H - pool) // stride + 1
        """
        N, C, H, W = x.shape
        p = self.pool_size
        s = self.stride
        oh = (H - p) // s + 1
        ow = (W - p) // s + 1

        out   = np.zeros((N, C, oh, ow), dtype=np.float32)
        masks = np.zeros_like(x, dtype=bool)

        for i in range(oh):
            for j in range(ow):
                patch = x[:, :, i*s:i*s+p, j*s:j*s+p]         # (N, C, p, p)
                max_v = patch.max(axis=(2, 3), keepdims=True)   # (N, C, 1, 1)
                out[:, :, i, j] = max_v[:, :, 0, 0]
                masks[:, :, i*s:i*s+p, j*s:j*s+p] |= (patch == max_v)

        self._cache = {"x_shape": x.shape, "masks": masks}
        return out

    def backward(self, dout):
        """
        Backward pass — route gradient only through the max element.

        dout : (N, C, OH, OW)
        Returns dx : (N, C, H, W)
        """
        x_shape = self._cache["x_shape"]
        masks   = self._cache["masks"]
        N, C, H, W = x_shape
        p = self.pool_size
        s = self.stride
        oh, ow = dout.shape[2], dout.shape[3]

        dx = np.zeros(x_shape, dtype=np.float32)
        for i in range(oh):
            for j in range(ow):
                window_mask = masks[:, :, i*s:i*s+p, j*s:j*s+p]
                d = dout[:, :, i, j][:, :, np.newaxis, np.newaxis]
                dx[:, :, i*s:i*s+p, j*s:j*s+p] += d * window_mask
        return dx


# ─────────────────────────────────────────────────────────────────────────────
# 5.  FLATTEN LAYER
# ─────────────────────────────────────────────────────────────────────────────

class FlattenLayer:
    """Reshapes (N, C, H, W) → (N, C·H·W)."""

    def __init__(self):
        self._cache = {}

    def forward(self, x):
        self._cache["shape"] = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, dout):
        return dout.reshape(self._cache["shape"])


# ─────────────────────────────────────────────────────────────────────────────
# 6.  FULLY CONNECTED LAYER
# ─────────────────────────────────────────────────────────────────────────────

class FCLayer:
    """
    Fully Connected (Dense) Layer with optional ReLU activation.

    Parameters
    ----------
    in_features  : number of input neurons
    out_features : number of output neurons
    activation   : 'relu' or None  (None → linear, used for output layer)
    """

    def __init__(self, in_features, out_features, activation=None):
        self.activation = activation

        scale = np.sqrt(2.0 / in_features)
        self.W = np.random.randn(in_features, out_features).astype(np.float32) * scale
        self.b = np.zeros((1, out_features), dtype=np.float32)

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        self._cache = {}

    def forward(self, x):
        """
        x : (N, in_features)
        Returns: (N, out_features)
        """
        z = x @ self.W + self.b          # linear combination
        self._cache["x"] = x
        self._cache["z"] = z

        if self.activation == "relu":
            out = relu(z)
        else:
            out = z
        return out

    def backward(self, dout):
        """
        dout : gradient w.r.t. layer output  (N, out_features)
        Returns dx : gradient w.r.t. layer input  (N, in_features)
        """
        x = self._cache["x"]
        z = self._cache["z"]

        if self.activation == "relu":
            dout = relu_backward(dout, z)

        self.dW = x.T @ dout
        self.db = dout.sum(axis=0, keepdims=True)
        dx = dout @ self.W.T
        return dx


# ─────────────────────────────────────────────────────────────────────────────
# 7.  LOSS — CROSS-ENTROPY WITH SOFTMAX
# ─────────────────────────────────────────────────────────────────────────────

def cross_entropy_loss(logits, y):
    """
    Computes softmax cross-entropy loss and gradient.

    logits : (N, C)  — raw scores from the final FC layer
    y      : (N,)    — integer class labels
    Returns loss (scalar), dlogits (N, C)
    """
    N = logits.shape[0]
    probs = softmax(logits)
    loss  = -np.log(probs[np.arange(N), y] + 1e-9).mean()
    dlogits        = probs.copy()
    dlogits[np.arange(N), y] -= 1
    dlogits       /= N
    return loss, dlogits


# ─────────────────────────────────────────────────────────────────────────────
# 8.  CNN MODEL
# ─────────────────────────────────────────────────────────────────────────────

class CNN:
    """
    Simple CNN:
        Conv(1→8, 3×3, pad=1) → ReLU → MaxPool(2×2)
        Conv(8→16, 3×3, pad=1) → ReLU → MaxPool(2×2)
        Flatten
        FC(784→128) → ReLU
        FC(128→10)  → (Softmax at loss)

    For 28×28 input:
        After conv1+pool1 : 8 × 14 × 14
        After conv2+pool2 : 16 × 7 × 7  = 784 features
    """

    def __init__(self):
        self.conv1   = ConvLayer(in_channels=1,  num_filters=8,  kernel_size=3, padding=1)
        self.pool1   = MaxPoolLayer(pool_size=2, stride=2)
        self.conv2   = ConvLayer(in_channels=8,  num_filters=16, kernel_size=3, padding=1)
        self.pool2   = MaxPoolLayer(pool_size=2, stride=2)
        self.flatten = FlattenLayer()
        self.fc1     = FCLayer(16 * 7 * 7, 128, activation="relu")
        self.fc2     = FCLayer(128, 10,         activation=None)

        self.layers  = [self.conv1, self.pool1, self.conv2, self.pool2,
                        self.flatten, self.fc1, self.fc2]

    # ── forward ──────────────────────────────────────────────────────────────

    def forward(self, x):
        """Full forward pass through all layers."""
        # Conv1 → ReLU → Pool1
        z1 = self.conv1.forward(x)       # pre-relu
        a1 = relu(z1)
        self._z1 = z1                    # cache for backward
        p1 = self.pool1.forward(a1)

        # Conv2 → ReLU → Pool2
        z2 = self.conv2.forward(p1)      # pre-relu
        a2 = relu(z2)
        self._z2 = z2                    # cache for backward
        p2 = self.pool2.forward(a2)

        # Flatten → FC1 (ReLU inside) → FC2 (linear logits)
        fl = self.flatten.forward(p2)
        h  = self.fc1.forward(fl)
        out = self.fc2.forward(h)
        return out

    # ── backward ─────────────────────────────────────────────────────────────

    def backward(self, dlogits):
        """Full backward pass (backpropagation) through all layers."""
        d = dlogits
        d = self.fc2.backward(d)
        d = self.fc1.backward(d)
        d = self.flatten.backward(d)

        # Through Pool2 and Conv2
        d = self.pool2.backward(d)
        d = relu_backward(d, self._z2)   # undo ReLU after conv2
        d = self.conv2.backward(d)

        # Through Pool1 and Conv1
        d = self.pool1.backward(d)
        d = relu_backward(d, self._z1)   # undo ReLU after conv1
        d = self.conv1.backward(d)
        return d

    # ── update (SGD) ─────────────────────────────────────────────────────────

    def update_params(self, lr):
        """Stochastic Gradient Descent weight update."""
        for layer in [self.conv1, self.conv2, self.fc1, self.fc2]:
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db


# ─────────────────────────────────────────────────────────────────────────────
# 9.  TRAINING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def train(model, X_train, y_train, epochs=5, batch_size=64, lr=0.01):
    """
    Training loop.

    Parameters
    ----------
    model      : CNN instance
    X_train    : (N, 1, 28, 28)  float32 in [0, 1]
    y_train    : (N,)            int labels
    epochs     : number of full passes over training data
    batch_size : mini-batch size
    lr         : learning rate
    """
    N = X_train.shape[0]
    for epoch in range(1, epochs + 1):
        indices = np.random.permutation(N)
        X_train = X_train[indices]
        y_train = y_train[indices]

        total_loss = 0.0
        n_batches  = 0

        for start in range(0, N, batch_size):
            xb = X_train[start:start + batch_size]
            yb = y_train[start:start + batch_size]

            # Forward pass
            logits = model.forward(xb)

            # Compute loss
            loss, dlogits = cross_entropy_loss(logits, yb)
            total_loss += loss
            n_batches  += 1

            # Backward pass
            model.backward(dlogits)

            # Update parameters
            model.update_params(lr)

            if n_batches % 100 == 0:
                print(f"  Epoch {epoch} | Batch {n_batches:4d}/{N//batch_size} "
                      f"| Loss: {loss:.4f}")

        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch}/{epochs} — Avg Loss: {avg_loss:.4f}")

    return model


# ─────────────────────────────────────────────────────────────────────────────
# 10. EVALUATION FUNCTION
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, X, y, batch_size=128):
    """
    Evaluate the model on the given data.

    Returns accuracy (float) and prints a per-class breakdown.
    """
    N = X.shape[0]
    all_preds = []

    for start in range(0, N, batch_size):
        xb = X[start:start + batch_size]
        logits = model.forward(xb)
        preds  = np.argmax(logits, axis=1)
        all_preds.append(preds)

    all_preds = np.concatenate(all_preds)
    accuracy  = (all_preds == y).mean() * 100

    print(f"\nOverall Accuracy: {accuracy:.2f}%")
    print(f"\n{'Class':<20} {'Correct':>8} {'Total':>8} {'Accuracy':>10}")
    print("-" * 50)
    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        mask    = y == cls_idx
        correct = (all_preds[mask] == y[mask]).sum()
        total   = mask.sum()
        cls_acc = 100 * correct / total if total > 0 else 0
        print(f"{cls_name:<20} {correct:>8} {total:>8} {cls_acc:>9.2f}%")
    print("-" * 50)

    return accuracy


# ─────────────────────────────────────────────────────────────────────────────
# 11. MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  CNN from Scratch — Fashion MNIST")
    print("=" * 60)

    # ── Load data ────────────────────────────────────────────────────────────
    print("\n[1] Loading Fashion MNIST …")
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    print(f"    Train : {X_train.shape}  Labels: {y_train.shape}")
    print(f"    Test  : {X_test.shape}   Labels: {y_test.shape}")

    # Use a subset for speed in demonstration; remove slice for full training
    TRAIN_SUBSET = 10000   # set to None or 60000 for full training
    if TRAIN_SUBSET:
        X_train = X_train[:TRAIN_SUBSET]
        y_train = y_train[:TRAIN_SUBSET]
        print(f"    (Using first {TRAIN_SUBSET} training samples for demo)")

    # ── Build model ──────────────────────────────────────────────────────────
    print("\n[2] Building CNN …")
    model = CNN()
    print("    Architecture:")
    print("      Conv(1→8,  3×3, pad=1) → ReLU → MaxPool(2×2)")
    print("      Conv(8→16, 3×3, pad=1) → ReLU → MaxPool(2×2)")
    print("      Flatten → FC(784→128, ReLU) → FC(128→10)")

    # ── Train ────────────────────────────────────────────────────────────────
    print("\n[3] Training …")
    model = train(model, X_train, y_train, epochs=5, batch_size=64, lr=0.01)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    print("\n[4] Evaluating on Test Set …")
    acc = evaluate(model, X_test, y_test)
    print(f"\nFinal Test Accuracy: {acc:.2f}%")
    print("\nDone.")
