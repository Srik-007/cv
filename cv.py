import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, GlobalAveragePooling2D
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.utils import to_categorical

# Create directory for saved models
os.makedirs("saved_models", exist_ok=True)

# Load FashionMNIST
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize and expand dimensions
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
x_train = np.expand_dims(x_train, -1)  # shape -> (num, 28, 28, 1)
x_test = np.expand_dims(x_test, -1)

# One-hot encode labels
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# Ask user to choose model
print("Select a model:")
print("1 - Basic CNN")
print("2 - ResNet50 (transfer learning)")
choice = input("Enter choice (1 or 2): ").strip()

# BASIC CNN
if choice == "1":
    model_name = "basic_cnn"
    model_path = f"saved_models/{model_name}.keras"

    if os.path.exists(model_path):
        print("🔁 Loading saved Basic CNN model...")
        model = load_model(model_path)
    else:
        print("🚀 Training Basic CNN...")
        model = Sequential([
            Input(shape=(28, 28, 1)),
            Conv2D(32, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(2, 2),
            Conv2D(64, (3, 3), activation='relu', padding='same'),
            MaxPooling2D(2, 2),
            Flatten(),
            Dense(128, activation='relu'),
            Dropout(0.5),
            Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train_cat, validation_data=(x_test, y_test_cat), epochs=5, batch_size=64, verbose=2)
        model.save(model_path)
        print("✅ Model saved.")

    _, acc = model.evaluate(x_test, y_test_cat, verbose=0)
    print(f"✅ Basic CNN Test Accuracy: {acc:.4f}")

# RESNET50 (Transfer Learning)
elif choice == "2":
    model_name = "resnet"
    model_path = f"saved_models/{model_name}.keras"

    def preprocess_resnet(image, label):
        image = tf.image.resize(image, [224, 224])
        image = tf.image.grayscale_to_rgb(image)
        return image, label

    # Prepare batched datasets with on-the-fly preprocessing
    train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train_cat)).map(preprocess_resnet).batch(64)
    test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test_cat)).map(preprocess_resnet).batch(64)

    if os.path.exists(model_path):
        print("🔁 Loading saved ResNet model...")
        model = load_model(model_path)
    else:
        print("🚀 Training ResNet50...")
        base_model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        base_model.trainable = False
        x = GlobalAveragePooling2D()(base_model.output)
        x = Dense(128, activation='relu')(x)
        output = Dense(10, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=output)

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(train_ds, validation_data=test_ds, epochs=1)
        model.save(model_path)
        print("✅ Model saved.")

    acc = model.evaluate(test_ds, verbose=0)
    print(f"✅ ResNet50 Test Accuracy: {acc:.4f}")

else:
    print("❌ Invalid choice. Please enter 1 or 2.")