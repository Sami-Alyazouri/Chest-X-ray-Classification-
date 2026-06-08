# 🫁 Chest X-Ray Classification — COVID-19 Detector

A deep learning model that classifies chest X-ray images into three diagnostic categories using a Convolutional Neural Network (CNN) built from scratch with TensorFlow and Keras.

---

## 📌 Problem Statement

Manual analysis of chest X-rays is time-consuming and prone to human error. This project automates the classification of X-ray images to assist in the early detection of COVID-19 and differentiate it from normal lungs and viral pneumonia.

---

## 🗂️ Classes

| Label | Description |
|-------|-------------|
| **COVID** | X-rays showing COVID-19 infection patterns |
| **Normal** | Healthy chest X-rays |
| **Viral Pneumonia** | X-rays showing viral pneumonia signs |

---

## 🧠 Model Architecture

```
Input (224 × 224 × 3)
        ↓
Conv2D(32, 3×3, ReLU, same) → MaxPooling2D(2×2)
        ↓
Conv2D(64, 3×3, ReLU, same) → MaxPooling2D(2×2)
        ↓
Conv2D(128, 3×3, ReLU, same) → MaxPooling2D(2×2)
        ↓
Flatten
        ↓
Dense(128, ReLU) → Dense(128, ReLU)
        ↓
Dense(3, Softmax)  ← Output
```

---

## ⚙️ Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Loss Function | Categorical Cross-Entropy |
| Epochs | 5 |
| Batch Size (Train) | 16 |
| Batch Size (Test) | 1 |
| Input Size | 224 × 224 × 3 |

---

## 📊 Results

### Training Progress

| Epoch | Train Accuracy | Train Loss | Val Accuracy | Val Loss |
|-------|---------------|------------|--------------|----------|
| 1 | 40.07% | 1.4185 | 63.64% | 0.7080 |
| 2 | 82.80% | 0.4485 | 84.85% | 0.5201 |
| 3 | 91.70% | 0.2301 | 84.85% | 0.3725 |
| 4 | 97.12% | 0.0967 | 89.39% | 0.2267 |
| 5 | 98.90% | 0.0347 | 89.39% | 0.3536 |

### Final Evaluation

| Split | Accuracy | Loss |
|-------|----------|------|
| **Test Set** | **87.35%** | 0.4004 |
| **Train Set** | **99.52%** | 0.0235 |

---

## 🛠️ Tools & Technologies

- **Python**
- **TensorFlow & Keras** — model building, training, evaluation
- **NumPy** — array manipulation and inference
- **Matplotlib** — loss and accuracy visualization
- **Kaggle** — cloud notebook environment and dataset

---

## 📁 Project Structure

```
xray-covid19-classifier/
│
├── X-Ray_classification_model.ipynb   # Main notebook
├── README.md                          # Project documentation
└── .gitignore
```

---

## 🚀 How to Run

1. Clone the repository:
```bash
git clone https://github.com/your-username/xray-covid19-classifier.git
```

2. Install dependencies:
```bash
pip install tensorflow keras numpy matplotlib
```

3. Download the dataset from [Kaggle — COVID-19 Image Dataset](https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset)

4. Run the notebook:
```bash
jupyter notebook X-Ray_classification_model.ipynb
```

---

## 📬 Contact

Feel free to reach out for questions or collaboration opportunities.
