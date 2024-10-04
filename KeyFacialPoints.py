import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense
from sklearn.model_selection import train_test_split

# Function to load the dataset
def load_data(file):
    df = pd.read_csv(file)
    return df

# Function to preprocess images
def preprocess_images(df):
    # Convert images from strings to numpy arrays
    images = df['Image'].apply(lambda x: np.fromstring(x, dtype=int, sep=' ').reshape(96, 96))
    images = images.values / 255.0  # Normalize images
    X = np.empty((len(images), 96, 96, 1))
    for i in range(len(images)):
        X[i,] = np.expand_dims(images[i], axis=2)
    return X

# Function to create and compile the model
def create_model(input_shape):
    X_input = Input(input_shape)
    X = Conv2D(64, (3, 3), activation='relu')(X_input)
    X = Flatten()(X)
    X = Dense(30, activation='relu')(X)
    model = Model(inputs=X_input, outputs=X)
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
    return model

# Function to plot key facial points
def plot_keypoints(image, keypoints):
    plt.imshow(image.reshape(96, 96), cmap='gray')
    plt.scatter(keypoints[::2], keypoints[1::2], c='red', s=10)  # Plot keypoints
    plt.axis('off')

# Streamlit app starts here
st.title('Facial Keypoints Detection')

# File upload
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])
if uploaded_file is not None:
    # Load and display data
    keyfacial_df = load_data(uploaded_file)
    st.write("Data Preview:")
    st.dataframe(keyfacial_df.head())

    # Preprocess images
    st.write("Processing images...")
    X = preprocess_images(keyfacial_df)

    # Targets
    y = keyfacial_df.iloc[:, :-1].values  # Assuming last column is image

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    st.write("Data split into training and testing sets.")

    # Create model
    input_shape = (96, 96, 1)
    model = create_model(input_shape)
    st.write("Model created.")

    # Train the model
    st.write("Training the model...")
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=50, batch_size=32)

    # Show training results
    st.write("Model training completed!")

    # Visualize loss and MAE
    st.subheader("Training and Validation Loss")
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    st.pyplot(plt)

    st.subheader("Training and Validation MAE")
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('MAE vs Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('MAE')
    plt.legend()
    st.pyplot(plt)

    # Make predictions
    st.write("Making predictions on test data...")
    y_pred = model.predict(X_test)

    # Display some sample images with key facial points
    st.subheader("Sample Images with Predicted Keypoints")
    sample_indices = np.random.choice(X_test.shape[0], 5, replace=False)
    for index in sample_indices:
        st.write(f"Image {index + 1}")
        fig, ax = plt.subplots()
        plot_keypoints(X_test[index], y_pred[index])
        st.pyplot(fig)

# Run the app using:
# streamlit run your_script_name.py
