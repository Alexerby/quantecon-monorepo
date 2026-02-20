import numpy as np
import matplotlib.pyplot as plt

from quantecon_lib.nn.layers import Dense
from quantecon_lib.nn.activations import ReLU, Sigmoid
from quantecon_lib.nn import FeedForwardNetwork
from quantecon_lib.nn.losses import MSE
from quantecon_lib.nn.optimizers import SGD
from quantecon_lib.viz.reconstruction import compare_reconstructions
from quantecon_lib.viz.core import use_quantecon_style

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

# 1. Load and Normalize Data
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data.astype('float32') / 255.0

# 2. Split into Train and Test (e.g., 80% train, 20% test)
# Since this is an Autoencoder, y is X
X_train, X_test = train_test_split(X[:10000], test_size=0.2, random_state=42)

# 3. Model Architecture
k = 43
layers = [
    Dense(784, 256), ReLU(),
    Dense(256, k),   ReLU(), # Bottleneck
    Dense(k, 256),   ReLU(),
    Dense(256, 784), Sigmoid()
]

model = FeedForwardNetwork(layers, SGD(learning_rate=0.1), MSE)

# Train with Test Set
# Passing X_test as both x_test and y_test for reconstruction validation
EPOCHS = 10 
model.train(
    x_train=X_train, 
    y_train=X_train, 
    x_test=X_test, 
    y_test=X_test, 
    epochs=EPOCHS,
    batch_size=32
)

# Visualize performance on the held-out test set
use_quantecon_style()

epochs_range = np.arange(len(model._history["training_loss"]))

compare_reconstructions(model, X_test[:5])

plt.plot(epochs_range, model._history["training_loss"], label="Training Loss", linewidth=4, alpha=0.5)
plt.plot(epochs_range, model._history["test_loss"], label="Test Loss", linestyle='--')
plt.legend()

plt.xlabel("Epochs")
plt.ylabel("MSE Loss")
plt.title("Learning Curve")

plt.show()
