import tensorflow as tf
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, Rescaling
from tensorflow.keras.models import Model
import os

print(" Initializing Pulmonology AI Training Protocol...")

# 1. Setup Image Parameters
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
BASE_DIR = 'chest_xray' 

# 2. Load Dataset (Automatically assigns NORMAL=0, PNEUMONIA=1)
print("Loading Chest X-Ray Dataset...")
train_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(BASE_DIR, 'train'),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

test_dataset = tf.keras.utils.image_dataset_from_directory(
    os.path.join(BASE_DIR, 'test'),
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

train_dataset = train_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# 3. Build the Correct Architecture (WITH PIXEL RESCALING)
inputs = tf.keras.Input(shape=(224, 224, 3))

# 👉 THIS LINE FIXES THE BUG: It converts 0-255 pixels into 0.0-1.0 decimals!
x = Rescaling(1./255)(inputs) 

# Load DenseNet and freeze it
base_model = DenseNet121(weights='imagenet', include_top=False)
base_model.trainable = False 

# Pass the scaled image to DenseNet
x = base_model(x, training=False)
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
predictions = Dense(1, activation='sigmoid')(x)

# Finalize Model
model = Model(inputs=inputs, outputs=predictions)

# 4. Compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# 👉 THIS FIXES THE CHEATING: Tell AI that NORMAL (0) is 3x more important than PNEUMONIA (1)
class_weights = {
    0: 3.0,  
    1: 1.0   
}

# 5. Train the AI (For 5 Epochs)
print(" Training starting... (Please wait while it learns to differentiate)")
model.fit(
    train_dataset,
    validation_data=test_dataset,
    epochs=5,
    class_weight=class_weights
)

# 6. Save the AI
os.makedirs("models", exist_ok=True)
model.save("models/densenet_lungs.h5")
print(" Pulmonology Model saved successfully at: models/densenet_lungs.h5")