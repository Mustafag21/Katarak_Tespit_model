import os
import numpy as np
import random
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# Veri Klasörleri
training_dir = "/Users/gulcu.21/Desktop/Dataset/train"
test_dir = "/Users/gulcu.21/Desktop/Dataset/test"

# Veri Boyutu
target_size = (128, 128)
batch_size = 32

# Veri Augmentasyonu
train_datagen = ImageDataGenerator(
    rescale=1 / 255.0,
    rotation_range=30,
    width_shift_range=0.3,
    height_shift_range=0.3,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    training_dir,
    target_size=target_size,
    class_mode='categorical',
    batch_size=batch_size
)

test_datagen = ImageDataGenerator(rescale=1 / 255.0)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    class_mode='categorical',
    batch_size=batch_size,
    shuffle=False
)

# MobileNetV2 Modeli
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Modelin üzerine yeni katmanlar ekle
x = base_model.output
x = Flatten()(x)
x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
x = Dropout(0.5)(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# İlk katmanları dondur
for layer in base_model.layers:
    layer.trainable = False

# Modeli derle
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6)

# Modeli eğit
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=test_generator,
    callbacks=[early_stopping, reduce_lr]
)

# Eğitim Sonuçlarını Görselleştir
plt.figure(figsize=(8, 6))
plt.plot(history.history['loss'], label='Train Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.plot(history.history['accuracy'], label='Train Accuracy', linestyle='--', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', linestyle='--', color='orange')
plt.title('Model Performance')
plt.xlabel('Epochs')
plt.ylabel('Metrics')
plt.legend()
plt.tight_layout()
plt.show()

# Test verileri üzerinde tahminler yap
test_generator.reset()
predictions = model.predict(test_generator, verbose=1)

# Tahminleri ve gerçek etiketleri al
predicted_classes = np.argmax(predictions, axis=1)
true_classes = test_generator.classes

# Karmaşıklık matrisini hesapla
cm = confusion_matrix(true_classes, predicted_classes)

# Karmaşıklık matrisini görselleştir
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=test_generator.class_indices.keys(), yticklabels=test_generator.class_indices.keys())
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()