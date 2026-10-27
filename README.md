# Facial Emotion Recognition Using Deep Learning

A deep learning project that classifies facial images into seven emotion categories using transfer learning, evaluates model performance, and provides an interactive prediction interface using Gradio.

## Project Overview

Facial Emotion Recognition (FER) is a computer vision task that identifies visual patterns associated with facial expressions. This project explores how pretrained convolutional neural networks can be adapted to classify facial images into multiple expression categories.

The project follows an end-to-end machine learning workflow, including exploratory data analysis, image preprocessing, class imbalance handling, transfer learning, model evaluation, Grad-CAM visualizations, and interactive inference.

**The best-performing model in the current experiments, EfficientNet-B0, achieved 70.56% validation accuracy.**

> **Important:** The system predicts visual expression categories, not a person's actual internal emotional state. Predictions may be incorrect and should not be treated as definitive psychological assessments.

## Objectives

- Explore and understand the distribution and characteristics of the facial image dataset.
- Apply image preprocessing and augmentation to prepare data for deep learning.
- Address class imbalance using class-weighted loss.
- Fine-tune and compare EfficientNet-B0 and MobileNetV3-Large.
- Evaluate performance using accuracy, precision, recall, F1-score, and confusion matrices.
- Inspect model predictions using Grad-CAM visualizations.
- Build an interactive Gradio interface for image-based inference.

## Dataset

The dataset used in this project contains facial images organized into seven emotion classes.

| Dataset split | Number of images |
|---|---:|
| Training | 28,831 |
| Validation | 7,066 |
| **Total** | **35,897** |

### Emotion categories

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

The images are 48 × 48 pixels and are processed into RGB format for compatibility with the pretrained models used in this project.

### Class imbalance

The dataset is imbalanced. The *happy* category accounts for approximately 24.85% of the training samples, while *disgust* accounts for approximately 1.51%.

Class-weighted CrossEntropyLoss was used to increase the contribution of underrepresented classes during training. This approach helps address imbalance but does not guarantee equal performance across categories.

*Dataset note:* The complete dataset is not included in this repository. Users should obtain it from the original source, review its licensing and usage conditions, and arrange the images into the expected training and validation directory structure before running the notebook.

## Technology Stack

| Area | Technologies |
|---|---|
| Programming language | Python |
| Development environment | Google Colab |
| Deep learning | PyTorch, Torchvision, timm |
| Image preprocessing | Albumentations, Pillow |
| Data analysis | NumPy, pandas |
| Visualization | Matplotlib, Seaborn |
| Model evaluation | scikit-learn |
| Interpretability | Grad-CAM |
| Interactive inference | Gradio |

## Methodology

### 1. Exploratory Data Analysis

The dataset was explored to understand class distribution, image dimensions, channel characteristics, pixel intensity patterns, and visual differences among emotion categories. PCA and t-SNE were also used to explore image representations.

### 2. Image Preprocessing and Augmentation

The preprocessing pipeline included:

- Resizing images to 224 × 224 pixels.
- Converting images to RGB.
- Applying horizontal flipping and other training-time augmentations.
- Applying brightness and contrast transformations.
- Normalizing images using ImageNet mean and standard deviation values.
- Converting images into PyTorch tensors.

Validation and inference preprocessing must remain consistent with the training pipeline.

### 3. Class Imbalance Handling

Class weights were calculated from the training labels and incorporated into CrossEntropyLoss. Label smoothing was also used to regularize the classification objective.

### 4. Transfer Learning

Two pretrained architectures were fine-tuned for seven-class emotion classification:

- **EfficientNet-B0:** A compact convolutional neural network used as the primary candidate model.
- **MobileNetV3-Large:** An efficient architecture included for performance comparison and potential resource-conscious deployment.

The final classification layers were adapted to match the seven target categories.

### 5. Model Evaluation

Both models were evaluated on the validation split using overall accuracy, per-class precision, recall and F1-score, classification reports, and confusion matrices.

Training and validation curves were also inspected to identify potential overfitting.

### 6. Model Interpretability

Grad-CAM was used to visualize image regions that contributed to selected model predictions. This supported qualitative inspection of model behavior but does not establish causal reasoning or psychological validity.

### 7. Interactive Inference

A Gradio interface was developed to allow users to upload a facial image, select a model, and view the predicted emotion and confidence score.

The interface provides a practical demonstration of the inference workflow. Interactive examples are functional checks, not a substitute for comprehensive model evaluation.

## Results

### Model comparison

| Metric | EfficientNet-B0 | MobileNetV3-Large |
|---|---:|---:|
| Validation accuracy | **70.56%** | 69.59% |
| Macro F1-score | 0.67 | 0.66 |
| Weighted F1-score | 0.71 | 0.70 |

### Key findings

- EfficientNet-B0 achieved the highest validation accuracy among the two evaluated models.
- Both models performed comparatively well on *happy* and *surprise*.
- *Angry*, *fear*, *sad*, and *neutral* showed more challenging classification patterns.
- The *disgust* category had limited representation, making its class-wise performance more difficult to interpret reliably.
- Training curves indicated a gap between training and validation performance, suggesting potential overfitting.

The accuracy difference between the two models is small. Further testing on an independent test set is necessary before drawing stronger conclusions about their relative performance.

## Project Structure

A suggested repository structure is:

```text
facial-emotion-recognition/
├── FER_project.ipynb
├── README.md
├── requirements.txt
├── app.py
├── class_names.txt
├── models/
│   ├── efficientnet_emotion.pth
│   └── mobilenetv3_emotion.pth
└── images/
    ├── class_distribution.png
    ├── confusion_matrix.png
    └── gradcam_examples.png
```

The `models/` and `images/` directories are optional. Include model weights and exported figures only when appropriate. Large model files may be better stored using Git LFS or a suitable model-hosting service.

The exact notebook and file names can be adjusted to match the files included in the repository. Do not include the dataset itself unless its licence explicitly permits redistribution.

## Getting Started

### Prerequisites

- Python 3.10 or another version compatible with the selected package versions.
- Access to the dataset in the expected folder structure.
- Google Colab or a local environment with sufficient resources for model training.
- Trained model checkpoints if you want to run inference without retraining.

### Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/facial-emotion-recognition.git
cd facial-emotion-recognition
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Replace `YOUR_USERNAME` with your GitHub username and adjust the repository name if needed.

### Run the notebook

Open `FER_project.ipynb` in Google Colab or Jupyter Notebook.

Configure the dataset paths to match your environment, execute the notebook cells in order, and train or evaluate the models as required.

### Run the Gradio application

If `app.py`, the trained model weights, class labels, and required dependencies are available, run:

```bash
python app.py
```

The application should start a Gradio server and provide a local URL. A public demonstration link requires a suitable hosting environment and remains subject to that service's availability and usage limits.

**Note:** The application architecture, checkpoint format, class ordering, and preprocessing must match the original training implementation. The notebook and application may require environment-specific path changes.

## Limitations

- Validation accuracy is approximately 70%, so prediction errors remain expected.
- Class imbalance affects the reliability of predictions for underrepresented categories.
- Similar visual patterns can lead to confusion between emotion classes.
- Validation results do not establish performance on unseen populations or real-world conditions.
- Lighting, pose, occlusion, image quality, and demographic variation may affect performance.
- Grad-CAM visualizations provide approximate explanations and are not proof of model reasoning.
- Softmax confidence scores are not necessarily calibrated probabilities of correctness.

## Future Improvements

- Evaluate on a separate, representative test dataset.
- Improve minority-class representation and assess class-wise performance.
- Experiment with augmentation strategies, learning-rate scheduling, and regularization.
- Compare additional pretrained architectures and tune hyperparameters systematically.
- Assess probability calibration and prediction reliability.
- Test robustness across lighting, pose, image quality, and demographic groups.
- Improve deployment reliability and add persistent logging where appropriate.

## Responsible Use

This project is intended for educational and experimental computer vision work. Facial expression classifications are imperfect visual predictions and cannot reliably establish what a person is feeling, thinking, or intending.

The model should not be used as the sole basis for consequential decisions involving employment, education, healthcare, law enforcement, or other sensitive settings. Use appropriate consent, privacy safeguards, and dataset licensing practices when handling facial images.

## Conclusion

This project demonstrates an end-to-end deep learning workflow for facial image classification, from data exploration and transfer learning to evaluation, interpretability, and interactive inference.

EfficientNet-B0 achieved 70.56% validation accuracy, slightly outperforming MobileNetV3-Large at 69.59%. The results establish a useful baseline while highlighting the need for improved generalization, minority-class evaluation, and independent testing.

The project provides practical experience with PyTorch, convolutional neural networks, imbalanced classification, model explainability, and interactive machine learning applications.

## Author

**Anagha P Nair**

Add your GitHub profile or professional portfolio link here.

## License

A licence has not yet been specified. Before adding one, review the dataset licence and any restrictions that apply to your code, pretrained weights, and derived artifacts. Do not assume that the dataset or model weights can be redistributed freely.
