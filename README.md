# Emotion AI

## Project Overview

Emotion AI, also known as Artificial Emotional Intelligence, is a branch of AI that enables machines to understand human emotions based on non-verbal cues such as body language and facial expressions. This project aims to classify people's emotions by analyzing their facial images.

The system employs deep learning techniques to predict facial keypoints and classify emotions. The project is divided into two parts: **Key Facial Points Detection** and **Emotion Detection**.

## Dataset

The project uses two datasets:

1. **Facial Keypoints Detection Dataset**:
   - **Source**: [Kaggle Facial Keypoints Detection Dataset](https://www.kaggle.com/c/facial-keypoints-detection/data)
   - **Details**: Contains x and y coordinates of 15 key facial points.
   - **Image Size**: 96x96 pixels, grayscale.

2. **Facial Expression Recognition Dataset**:
   - **Source**: [Kaggle Emotion Detection Dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
   - **Details**: Images categorized into 5 emotions: Angry, Disgust, Sad, Happy, Surprise.
   - **Image Size**: 48x48 pixels, grayscale.

## Model Architecture

1. **Key Facial Points Detection Model**:
   - Built using a **Convolutional Neural Network (CNN)** with **Residual Blocks** to predict facial keypoints.
   - The model outputs the coordinates of 15 key points on the face.

2. **Emotion Detection Model**:
   - A deep learning classifier is used to categorize the input image into one of five emotion classes.
   - Emotions classified are: Angry, Disgust, Sad, Happy, Surprise.

Both models are integrated to produce combined predictions of facial key points and emotions from a given image.

## Results

The models are evaluated using performance metrics like **accuracy**, **precision**, and **recall**. A confusion matrix is used to illustrate the classification results.

### Sample Result:
- **Emotion Prediction**: Happiness
- **Facial Keypoints**: Predicted with x, y coordinates of 15 key points.

## Technologies Used

- **Python**
- **TensorFlow**: For model training and serving.
- **Keras**: High-level neural network API.
- **OpenCV**: For image preprocessing.
- **Matplotlib**: For visualizing training results.

## Acknowledgments

Special thanks to the teams that provided the datasets used in this project:

- [Kaggle Facial Keypoints Detection Dataset](https://www.kaggle.com/c/facial-keypoints-detection/data)
- [Kaggle Emotion Detection Dataset](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge/data)
