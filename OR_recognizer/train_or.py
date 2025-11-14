import os
import numpy as np
from neuron import Neuron

# OR gate dataset
X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
])
y = np.array([0.0, 1.0, 1.0, 1.0])  # expected outputs

np.random.seed(42)  # for reproducibility
neuron = Neuron(n_inputs=2, weight_scale=0.1)

# Training loop
n_epochs = 2000
learning_rate = 0.1

# Prepare output file next to this script
out_path = os.path.join(os.path.dirname(__file__), "training_output.txt")

def _format(msg: str) -> str:
    return str(msg)

with open(out_path, "w", encoding="utf-8") as out:
    def log(msg: str) -> None:
        """Print to console and write to the output file."""
        s = _format(msg)
        print(s)
        out.write(s + "\n")

    for epoch in range(1, n_epochs + 1):
        epoch_loss = 0.0

        for x_i, yi in zip(X, y):
            loss = neuron.train_on_example(x_i, yi, lr=learning_rate)
            epoch_loss += loss
        log(f"Epoch {epoch}, Loss: {epoch_loss:.4f}")

        epoch_loss /= len(X)

        if epoch == 1 or epoch % 200 == 0:
            log(f"Epoch {epoch}, Average Loss: {epoch_loss:.6f}")

    log("\nTrained weights: " + np.array2string(neuron.w, precision=6))
    log("Trained bias: " + f"{neuron.b:.6f}")

    log("\nPredictions after training:")
    for x_i, y_i in zip(X, y):
        pred = neuron.forward(x_i)
        log(f"Input: {x_i}, Predicted: {pred:.4f}, Expected: {y_i}")

    log(f"\nOutput saved to: {out_path}")