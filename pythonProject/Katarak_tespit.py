import os
import numpy as np
import random
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tensorflow.keras.preprocessing.image import ImageDataGenerator



test_dir = "/Users/gulcu.21/Desktop/Dataset/test" #Test verilerinin dosya
training_dir = "/Users/gulcu.21/Desktop/Dataset/train" # Eğitim verilerinin dosya yolu
target_size = (128, 128) #Görüntülerin Belirli Bir Ölçekte Yeniden Boyutlandırılması

model_path = "trained_mobilenet_model3.h5" # Eğer model daha önceden eğitilmiş ise modeli yüklemek için modelin yolu

if not os.path.exists(model_path): # eğer model daha önceden eğitilip kaydedilmemiş ise model eğitim bloğuna girer
    # MODEL EĞİTİMİ İÇİN TENSORFLOW KÜTÜPHANESİ VE FONKSİYONLARINI İMPORT ETMEK
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Dense, Dropout, Flatten
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    #Görüntülerin yeniden ölçeklendirilmesi ve çeşitli dönüşümlerle artırılması (rotasyon, kaydırma, yakınlaştırma, vb.)
    train_datagen = ImageDataGenerator( #görüntülerin augmentasyonu (çeşitlendirilmesi) ve ön işlenmesi için bir araçtır.
        rescale=1 / 255.0, #Görüntü piksel değerlerini [0, 255] aralığından [0, 1] aralığına dönüştürür.
        rotation_range=30, #Görüntüleri rastgele 30 dereceye kadar döndürür.
        width_shift_range=0.3,#Görüntüyü yatayda %30 oranında kaydırır
        height_shift_range=0.3,#Görüntüyü dikeyde %30 oranında kaydırır
        shear_range=0.3, #Görüntülere %30 oranında kayma (shear) uygular.
        zoom_range=0.3, #Görüntüyü %30 oranında yakınlaştırır veya uzaklaştırır.
        horizontal_flip=True #Görüntüyü yatay olarak çevirir (flip).
    )
    #Bu fonksiyon, görüntülerin hedef boyuta getirilmesini ve augmentasyon işlemlerinin uygulanmasını sağlar.
    train_generator = train_datagen.flow_from_directory( #
        training_dir, #veri setini training_dir klasöründen yükler.
        target_size=target_size,#Görüntüler belirtilen boyuta (128x128 piksel) yeniden ölçeklendirilir.
        class_mode='categorical', # Sınıf etiketleri kategorik bir formatta kodlanır (ör. [1, 0, 0] gibi bir one-hot vektörü).
        batch_size=32 # Görüntüler 32'lik mini-batch'ler halinde yüklenir model bu nedenle her defasında 32 görüntü ile çalışır
    )

    test_datagen = ImageDataGenerator(rescale=1 / 255.0) #Test verileri için yalnızca ölçekleme işlemi yapılır
    test_generator = test_datagen.flow_from_directory(
        test_dir,
        target_size=target_size,
        class_mode='categorical',
        batch_size=32,
        shuffle=False
    )

    # MobileNetV2 Modeli
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

    # Modelin üzerine yeni katmanlar ekle
    #Modelin üzerine tam bağlantılı (fully connected) ve dropout katmanları eklenerek, iki sınıfı (binary classification) tahmin etmek için yapılandırılmıştır.
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
        optimizer=Adam(learning_rate=0.0001),#Parametre güncellemesi için optimize edilmiş bir algoritma.
        loss='categorical_crossentropy',#Çok sınıflı sınıflandırma için kayıp fonksiyonu.
        metrics=['accuracy']#Modelin performansını ölçmek için doğruluk metriği.
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
    model.save(model_path)
    print("Model eğitildi ve kaydedildi.")
else:
    # Eğitilmiş modeli yükle
    model = load_model(model_path)
    print("Eğitilmiş model yüklendi.")

# Eğitim süreci doğruluk ve kayıp grafiklerini çiz
if 'history' in locals() or 'history' in globals():
    plt.figure(figsize=(12, 6))

    # Doğruluk grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Kayıp grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Test veri kümesinde tahmin yap
test_datagen = ImageDataGenerator(rescale=1 / 255.0)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=target_size,
    class_mode='categorical',
    batch_size=32,
    shuffle=False
)

true_labels = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# Tahminler
predictions = model.predict(test_generator)
y_pred = np.argmax(predictions, axis=1)

# Karmaşıklık Matrisi
conf_matrix = confusion_matrix(true_labels, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# Rastgele 10 Test Görüntüsü Tahmini ve Görselleştirme
def predict_and_show_images(model, test_dir, target_size=(128, 128), n_images=10):
    images = []
    image_paths = []
    categories = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    class_indices = {category: idx for idx, category in enumerate(categories)}
    class_labels = list(class_indices.keys())

    # Rastgele 10 görüntü seç
    for _ in range(n_images):
        random_category = random.choice(categories)
        category_path = os.path.join(test_dir, random_category)
        random_image = random.choice(os.listdir(category_path))
        image_path = os.path.join(category_path, random_image)

        img = Image.open(image_path)
        img_resized = img.resize(target_size)
        img_array = np.array(img_resized) / 255.0
        if len(img_array.shape) == 2:
            img_array = np.stack((img_array,) * 3, axis=-1)
        images.append(img_array)
        image_paths.append(image_path)

    # Tahmin yap
    images = np.array(images)
    predictions = model.predict(images)

    # Tahminleri görselleştir
    for i, pred in enumerate(predictions):
        class_idx = np.argmax(pred)
        class_label = class_labels[class_idx]
        confidence = pred[class_idx]
        plt.figure(figsize=(8,6))
        img = Image.open(image_paths[i])
        plt.imshow(img)
        plt.title(f"Model Tahmin: {class_label} ({confidence:.2f})\n Görüntü Yolu: {image_paths[i]}")
        plt.axis('off')
        plt.show()

predict_and_show_images(model, test_dir, target_size=target_size, n_images=10)