# AnomalyDetection_Autoencoder
Anomaly detection using autoencoders and CNNs implemented in Python with TensorFlow/Keras.
# Anomaly Detection in Manufacturing Using Autoencoders

## Project Overview
This project implements an autoencoder-based anomaly detection system for quality control in manufacturing environments. Using the **MVTec dataset**, we detect and classify anomalies in defective images of one chosen object: **toothbrushes**. The system determines whether an image represents a defective or good product.

The project evaluates three autoencoder architectures:
- Dense Neural Network with ReLU activation
- Dense Neural Network with Leaky ReLU activation
- Convolutional Neural Network (CNN)

Our best models achieved:
- **97.62% Accuracy**
- **96.77% Precision**
- **100% Recall**

These results highlight the effectiveness of deep learning for automated quality assurance.

---

## Features
- **Data Preprocessing:** Converts images to grayscale, resizes them to 128x128 pixels, normalizes, and reshapes them for model compatibility.
- **Anomaly Detection:** Measures reconstruction error from autoencoders to classify defective and good products.
- **Model Comparison:** Implements and compares Dense and CNN autoencoder architectures.
- **Evaluation Metrics:** Reports accuracy, precision, recall, and F1 score.
- **Reconstruction Visualization:** Allows visual analysis of reconstructed images for insight into model behavior.

---

## Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/anomaly-detection-autoencoder.git
    ```
2. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Open the Jupyter Notebook:
    ```bash
    jupyter notebook DVA263_PRO1_Group_6.ipynb
    ```

---

## Dataset
The **MVTec Anomaly Detection dataset** is used, focusing on the toothbrush class. The dataset includes:
- **Training Set:** Images of non-defective products
- **Test Set:** Images of both defective and non-defective products
- **Ground Truth:** Labels for defective samples

Dataset preprocessing involves:
- Grayscale conversion
- Resizing to 128x128
- Normalization to [0,1]

---

## Model Architectures
### 1. Dense ReLU Autoencoder
```python
Encoder:
- Dense(1024, activation='relu')
- Dense(512, activation='relu')
- Dense(256, activation='relu')
- Dense(128, activation='relu')

Decoder:
- Dense(256, activation='relu')
- Dense(512, activation='relu')
- Dense(1024, activation='relu')
- Dense(input_shape, activation='sigmoid')
```

### 2. Dense Leaky ReLU Autoencoder
Similar to Dense ReLU but replaces ReLU with Leaky ReLU (alpha=0.01).

### 3. Convolutional Neural Network (CNN) Autoencoder
```python
Encoder:
- Conv2D(32, kernel_size=(3,3), activation='relu')
- MaxPooling2D(pool_size=(2,2))
- Conv2D(64, kernel_size=(3,3), activation='relu')
- MaxPooling2D(pool_size=(2,2))
- Conv2D(128, kernel_size=(3,3), activation='relu')
- MaxPooling2D(pool_size=(2,2))

Decoder:
- Conv2DTranspose(128, kernel_size=(3,3), activation='relu')
- UpSampling2D(size=(2,2))
- Conv2DTranspose(64, kernel_size=(3,3), activation='relu')
- UpSampling2D(size=(2,2))
- Conv2DTranspose(32, kernel_size=(3,3), activation='relu')
- UpSampling2D(size=(2,2))
- Conv2D(1, kernel_size=(3,3), activation='sigmoid')
```

---

## Evaluation
### Metrics:
- **Accuracy:** 97.62%
- **Precision:** 96.77%
- **Recall:** 100%
- **F1 Score:** 98.36%

### Key Findings:
- **Dense ReLU and CNN models** outperformed Leaky ReLU in all metrics.
- Both Dense ReLU and CNN achieved **perfect recall**, ensuring no defective products were missed.

---

## Usage
### Training:
1. Preprocess the dataset.
2. Train any of the three autoencoder architectures provided.

### Testing:
1. Evaluate the model on defective and non-defective images.
2. Use reconstruction error thresholding for classification.

---

## Results
### Reconstruction Quality
- **Dense ReLU:** Preserves global structure well.
- **CNN:** Retains finer details better.
- **Leaky ReLU:** Sensitive to noise, underperforms compared to other models.

### Anomaly Detection
Defects detected using reconstruction error thresholds derived from non-defective samples.

---

## Future Work
- Incorporate ensemble methods for better performance.
- Explore attention mechanisms for focused anomaly detection.
- Optimize thresholding techniques for unsupervised settings.
- Extend to real-time and multi-product quality control.

---

## Authors
- Aya Mohammad and Zaina Hamdan - Group6 (DVA263 Course, Mälardalen University)

---

## License
This project is licensed under the [MIT License](LICENSE).

---

## References
1. Bergmann et al. (2019). Improving unsupervised defect segmentation by applying structural similarity to autoencoders.
2. Chandola et al. (2009). Anomaly detection: A survey.
3. Chalapathy & Chawla (2019). Deep learning for anomaly detection: A survey.
4. Masci et al. (2011). Stacked convolutional auto-encoders for hierarchical feature extraction.
5. Ruff et al. (2018). Deep one-class classification.
6. Zhou & Paffenroth (2017). Anomaly detection with robust deep autoencoders.

