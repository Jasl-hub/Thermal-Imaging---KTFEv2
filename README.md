# ThermoEmo: Enhanced Facial Emotion Recognition via Thermal Imaging and Deep Learning

## Overview
This repository contains the implementation of a deep learning model for facial emotion recognition using thermal imaging. The work is based on the research presented at the **IEEE CALCON Conference 2024** titled *"Enhanced Facial Emotion Recognition via Thermal Imaging and Deep Learning: KTFEv2 Study"* (DOI: [10.1109/CALCON63337.2024.10914333](https://doi.org/10.1109/CALCON63337.2024.10914333)).

## Features
- **Thermal Image Processing**: Utilizes infrared thermal images for enhanced emotion detection.
- **Deep Learning Architecture**: Implements a CNN-based approach for feature extraction and classification.
- **7-Class Emotion Classification**: Recognizes emotions: *Anger, Disgust, Surprise, Neutral, Fear, Sad, and Happy*.
- **Data Augmentation & Preprocessing**: Includes resizing, normalization, and dataset handling.
- **Training & Evaluation**: Implements train-validation split, accuracy tracking, and visualization tools.
- **Confusion Matrix & Classification Report**: Provides model performance insights.

## Dataset
The dataset used in this study consists of thermal images categorized into seven emotion classes. The dataset is extracted from `/content/EM/thermal/` and preprocessed before training.

## Model Architecture
The CNN model consists of:
1. Convolutional layers (12, 20, 128 filters)
2. MaxPooling layers
3. Dropout layers for regularization
4. Fully connected Dense layers
5. Softmax activation for classification

## Installation & Dependencies
To set up the project, install the required dependencies:
```bash
pip install numpy pandas opencv-python keras tensorflow matplotlib scikit-learn
```

## How to Run
1. Mount Google Drive and unzip dataset:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !unzip /content/drive/MyDrive/thermal.zip -d '/content/EM'
   ```
2. Preprocess images and prepare data.
3. Train the model using:
   ```python
   model.fit(X_train, y_train, batch_size=7, epochs=50, validation_data=(X_test, y_test))
   ```
4. Evaluate the model:
   ```python
   score = model.evaluate(X_test, y_test)
   print('Test Accuracy:', score[1])
   ```
5. Visualize results with confusion matrix and accuracy plots.

## Results
- **Accuracy Achieved**: The model achieves competitive accuracy in emotion recognition.
- **Confusion Matrix**: Displays class-wise performance.
- **Classification Report**: Precision, Recall, and F1-score metrics for evaluation.

## Citation
If you use this work, please cite the IEEE CALCON Conference 2024 paper:
```
@inproceedings{KTFEv2_2024,
  author={Jasleen Kaur},
  title={Enhanced Facial Emotion Recognition via Thermal Imaging and Deep Learning: KTFEv2 Study},
  booktitle={IEEE CALCON Conference 2024},
  year={2024},
  doi={10.1109/CALCON63337.2024.10914333}
}
```

## Contact
For queries or collaboration, contact: www.linkedin.com/in/jas03leen

## License
This project is for research purposes. Please refer to the conference publication for more details.

