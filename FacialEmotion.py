import streamlit as st
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, ZeroPadding2D, Conv2D, BatchNormalization, Activation, MaxPooling2D, AveragePooling2D, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

st.set_page_config(page_title="Facial Emotion Recognition", layout="wide")

# Add a sidebar with options
st.sidebar.title("Facial Emotion Recognition")
st.sidebar.write("Explore different options below:")

# Sidebar options
if st.sidebar.checkbox('Show Data Description'):
    st.markdown("""
    ### Facial Emotion Recognition App
    This app helps to detect emotions from facial images using a Convolutional Neural Network (CNN).
    The emotions are categorized as:
    - Angry
    - Disgust
    - Sad
    - Happy
    - Surprise
    """)

# Function Definitions
def string2array(x):
    if isinstance(x, str):
        return np.array(x.split(' ')).reshape(48, 48, 1).astype('float32')
    else:
        return x

def resize(x):
    if x.shape == (48, 48, 1):
        img = x.reshape(48, 48)
        return cv2.resize(img, dsize=(96, 96), interpolation=cv2.INTER_CUBIC)
    else:
        return x

# Load Data
@st.cache_data
def load_data():
    df = pd.read_csv('icml_face_data.csv')
    df[' pixels'] = df[' pixels'].apply(lambda x: string2array(x))
    df[' pixels'] = df[' pixels'].apply(lambda x: resize(x))
    return df

# Load data with spinner
with st.spinner('Loading data...'):
    facialexpression_df = load_data()
    st.sidebar.success("Data loaded successfully!")

# Prepare Data
X = np.stack(facialexpression_df[' pixels'], axis=0)
y = to_categorical(facialexpression_df['emotion'])
X = X.reshape(X.shape[0], 96, 96, 1) / 255.0

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, shuffle=True)

# Data Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Define Model
def create_model():
    input_shape = (96, 96, 1)
    X_input = Input(input_shape)
    X = ZeroPadding2D((3, 3))(X_input)
    X = Conv2D(64, (7, 7), strides=(2, 2), kernel_initializer='glorot_uniform')(X)
    X = BatchNormalization(axis=3)(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)
    X = AveragePooling2D((4, 4))(X)
    X = Flatten()(X)
    X = Dense(5, activation='softmax')(X)
    model = Model(inputs=X_input, outputs=X)
    return model

model_2_emotion = create_model()
model_2_emotion.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Model Training
st.title("Facial Emotion Recognition System")
st.write("This app helps to train a deep learning model to detect emotions from facial images.")

col1, col2 = st.columns(2)

with col1:
    if st.button('Train Model'):
        earlystopping = EarlyStopping(monitor='val_loss', mode='min', patience=20)
        checkpointer = ModelCheckpoint(filepath="FacialExpression_weights.keras", verbose=1, save_best_only=True)
        with st.spinner("Training the model..."):
            history = model_2_emotion.fit(train_datagen.flow(X_train, y_train, batch_size=64),
                                        validation_data=(X_val, y_val),
                                        steps_per_epoch=len(X_train) // 64,
                                        epochs=20,
                                        callbacks=[checkpointer, earlystopping])
            st.success("Model Training Completed!")
        st.balloons()

# Model Evaluation
with col2:
    if st.button('Evaluate Model'):
        score = model_2_emotion.evaluate(X_test, y_test)
        st.write(f'Test Accuracy: {score[1]:.2f}')
        st.success("Model Evaluation Completed!")

# Prediction and Visualization
if st.button('Make Predictions'):
    predicted_classes = np.argmax(model_2_emotion.predict(X_test), axis=-1)
    y_true = np.argmax(y_test, axis=-1)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, predicted_classes)
    plt.figure(figsize=(10, 10))
    sns.heatmap(cm, annot=True, cbar=False)
    st.pyplot(plt)
    st.info("Confusion matrix visualization complete!")

# Emotion Mapping
emotion_labels = {
    0: "Angry",
    1: "Disgust",
    2: "Sad",
    3: "Happy",
    4: "Surprise",
}

# Display sample images
if st.button('Show Sample Images with Predictions'):
    num_images = 10  # Number of images to display
    indices = np.random.choice(len(X_test), num_images, replace=False)  # Randomly select indices

    for i in indices:
        img_to_display = X_test[i].reshape(96, 96)
        img_to_display = np.clip(img_to_display, 0, 1)

        predicted_emotion = np.argmax(model_2_emotion.predict(X_test[i].reshape(1, 96, 96, 1)), axis=-1)
        emotion_label = emotion_labels[predicted_emotion[0]]  # Get the predicted emotion tag

        # Layout for images and predictions
        st.image(img_to_display, caption=f'Predicted Emotion: {emotion_label}', channels='GRAY', use_column_width=True)
