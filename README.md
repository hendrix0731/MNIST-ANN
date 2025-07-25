# ✍️ Handwritten Digit Recognition with ANN (MNIST Dataset)

This project implements an **Artificial Neural Network (ANN)** using **Keras** to classify handwritten digits (0–9) from the **MNIST dataset**. It’s a classic deep learning task demonstrating how feedforward neural networks can be applied to image classification.

---

## 🗃 Dataset

- **Name:** MNIST – Modified National Institute of Standards and Technology
- **Source:** [`tensorflow.keras.datasets.mnist`](https://keras.io/api/datasets/mnist/)
- **Size:** 70,000 grayscale images (60,000 train + 10,000 test)
- **Image Size:** 28×28 pixels
- **Classes:** 10 (digits 0 to 9)

---

## 🧠 Model Architecture

This model uses a basic **fully connected feedforward neural network (ANN)** with `relu` activations and a `softmax` output layer for classification.

```python
Input: 784 (28x28 flattened)
Dense(128, activation='relu')
Dense(64, activation='relu')
Dense(10, activation='softmax')
```

---

## ⚙️ Technologies Used

- Python 3.11
- TensorFlow / Keras
- NumPy, Matplotlib
- Jupyter Notebook

---

## 🚀 How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/hendrix0731/MNIST-ANN.git
   cd MNIST-ANN
   ```

2. Install dependencies:
   ```bash
   pip install tensorflow numpy matplotlib
   ```

3. Run the notebook:
   Open `mnistdataset.ipynb` in Jupyter or any IDE and run all cells.

---

## 📈 Results

> Typical results after 10–15 epochs:

- **Training Accuracy:** ~98%
- **Test Accuracy:** ~97%
- **Loss:** converges well with minimal overfitting

📌 *Results may vary depending on the number of epochs, batch size, and initialization.*

---

```python
import matplotlib.pyplot as plt
plt.imshow(x_test[0].reshape(28,28), cmap='gray')
print("Predicted Label:", model.predict(x_test[0].reshape(1,784)).argmax())
```

---

## 📦 Save and Load Model

```python
# Save model
model.save("mnist_ann_model.h5")

# Load model
from keras.models import load_model
model = load_model("mnist_ann_model.h5")
```

---

## 👨‍💻 Author

**Harsh Joshi**  
GitHub: [@hendrix0731](https://github.com/hendrix0731)

---

## 📜 License

This project is open-source under the [MIT License](LICENSE).

