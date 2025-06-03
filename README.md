# DeepvisionCNNvsDNN

This project demonstrates image classification using two popular datasets: **MNIST** (handwritten digits) and **Fashion MNIST** (clothing items), comparing the performance of a Convolutional Neural Network (CNN) and a basic Deep Neural Network (DNN). Built using TensorFlow and Keras, this project showcases how deep learning can be applied to grayscale image data.

## Project Summary
- **Dataset Size**: 70,000 images per dataset (28x28 grayscale)
- **Models Used**:
  - Deep Neural Network (DNN)
  - Convolutional Neural Network (CNN)
- **Key Techniques**:
  - Data preprocessing (normalization, reshaping)
  - One-hot encoding of labels
  - Use of validation sets
  - Early stopping to avoid overfitting

## Results

| Dataset        | Model | Accuracy |
|----------------|-------|----------|
| MNIST          | DNN   | ~96.2%   |
| MNIST          | CNN   | ~98.3%   |
| Fashion MNIST  | DNN   | ~85.1%   |
| Fashion MNIST  | CNN   | ~89.7%   |

CNN outperformed DNN on both datasets, especially on the more complex Fashion MNIST images.

## Getting Started

### Prerequisites

Make sure you have the following installed:

- Python 3.7+
- TensorFlow 2.x
- Keras (integrated in TensorFlow)
- Jupyter Notebook

### Installation

```bash
git clone https://github.com/aryan-b-shah/DeepvisionCNNvsDNN.git
cd DeepvisionCNNvsDNN
```

### Run the Notebook
```bash
jupyter notebook DeepvisionCNNvsDNN.ipynb
```

## Model Architecture

### DNN:

- Input Layer (784)
- Dense (128, ReLU)
- Dense (64, ReLU)
- Output Layer (10, Softmax)

### CNN:

- Conv2D (32 filters, 3x3, ReLU)
- MaxPooling2D
- Conv2D (64 filters, 3x3, ReLU)
- MaxPooling2D
- Flatten
- Dense (64, ReLU)
- Output Layer (10, Softmax)

## Highlights

- Compared two types of neural networks to illustrate the effectiveness of CNNs in image-based tasks.
- Achieved high accuracy using simple model tuning and training strategies.
- Demonstrated generalization across two distinct datasets without extensive hyperparameter tuning.
