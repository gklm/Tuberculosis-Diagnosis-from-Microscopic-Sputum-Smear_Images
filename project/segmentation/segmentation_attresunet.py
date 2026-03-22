# Import necessary libraries
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input, Conv2D, MaxPooling2D, UpSampling2D, Concatenate, BatchNormalization, Activation)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Load and preprocess data (modify this section to use your own data)
# Placeholder paths for dataset
image_folder = 'path/to/images/'
mask_folder = 'path/to/masks/'

# Function to load images
def load_images(folder):
    images = []
    for filename in os.listdir(folder):
        img = plt.imread(os.path.join(folder, filename))
        images.append(img)
    return np.array(images)

# Load images and masks
images = load_images(image_folder)
masks = load_images(mask_folder)

# Normalize images
images = images / 255.0
masks = masks / 255.0

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, masks, test_size=0.2, random_state=42)

# Define Attention-Residual U-Net model
def attention_residual_unet(input_shape):
    inputs = Input(input_shape)
    
    # Encoder
    c1 = Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = BatchNormalization()(c1)
    p1 = MaxPooling2D((2, 2))(c1)
    
    c2 = Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = BatchNormalization()(c2)
    p2 = MaxPooling2D((2, 2))(c2)

    # Bottleneck
    bn = Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    bn = BatchNormalization()(bn)

    # Decoder
    u1 = UpSampling2D((2, 2))(bn)
    u1 = Concatenate()([u1, c2])
    c3 = Conv2D(128, (3, 3), activation='relu', padding='same')(u1)
    c3 = BatchNormalization()(c3)

    u2 = UpSampling2D((2, 2))(c3)
    u2 = Concatenate()([u2, c1])
    c4 = Conv2D(64, (3, 3), activation='relu', padding='same')(u2)
    c4 = BatchNormalization()(c4)

    outputs = Conv2D(1, (1, 1), activation='sigmoid')(c4)

    model = Model(inputs, outputs)
    return model

# Compile the model
model = attention_residual_unet(input_shape=(128, 128, 3))
model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

# Define callbacks
checkpoint = ModelCheckpoint('attresunet_model.h5', save_best_only=True, monitor='val_loss', mode='min')
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6)

# Train the model
epochs = 50
batch_size = 16
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=epochs,
    batch_size=batch_size,
    callbacks=[checkpoint, reduce_lr]
)

# Plot training history
plt.plot(history.history['accuracy'], label='train accuracy')
plt.plot(history.history['val_accuracy'], label='val accuracy')
plt.legend()
plt.show()
