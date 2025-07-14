# Handwritten Digit Recognition using CNN

This project implements a Convolutional Neural Network (CNN) to recognize handwritten digits (0–9) using the MNIST dataset in CSV format.

---

## 📂 Dataset

- **Source:** [MNIST CSV version](https://www.kaggle.com/datasets/oddrationale/mnist-in-csv)
- **Structure:** Each row represents a 28x28 grayscale image (784 pixels) with a label (0–9)

---

## 🚀 Project Workflow

1. **Load CSV**:
   - Used `pandas` to load the MNIST CSV file.

2. **Preprocessing**:
   - Split into features (`X`) and labels (`y`)
   - Normalized pixel values to range [0, 1]
   - Reshaped `X` to `(samples, 28, 28, 1)` for CNN input

3. **Model Architecture**:
   ```python
   model = Sequential([
       Conv2D(32, (3,3), activation='relu'),
       MaxPooling2D(2,2),
       Conv2D(64, (3,3), activation='relu'),
       MaxPooling2D(2,2),
       Flatten(),
       Dense(64, activation='relu'),
       Dense(10, activation='softmax')
   ])
