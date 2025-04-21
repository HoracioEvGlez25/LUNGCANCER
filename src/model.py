import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing import image
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
import numpy as np
import os
from collections import Counter


def load_images():
    images = []
    labels = []

    image_dirs = {
        'benigno': 'C:/Users/juane/OneDrive/Desktop/cancer_lung_prediction/data/imagenes/benigno',
        'maligno': 'C:/Users/juane/OneDrive/Desktop/cancer_lung_prediction/data/imagenes/maligno',
        'normal':  'C:/Users/juane/OneDrive/Desktop/cancer_lung_prediction/data/imagenes/normal'
    }

    for label, dir_path in image_dirs.items():
        for filename in os.listdir(dir_path):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                img_path = os.path.join(dir_path, filename)
                img = image.load_img(img_path, target_size=(224, 224))
                img_array = image.img_to_array(img) / 255.0
                images.append(img_array)
                labels.append(label)

    images = np.array(images)

    # Codificar etiquetas: benigno=0, maligno=1, normal=2
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)

    return images, labels, label_encoder


def create_model():
    print("Creando el modelo...")
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))  # 3 clases

    model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def split_data_by_class(images, labels, test_size=0.2):
    images = np.array(images)
    labels = np.array(labels)

    X_train, X_val, y_train, y_val = [], [], [], []

    for class_id in np.unique(labels):
        class_indices = np.where(labels == class_id)[0]
        class_images = images[class_indices]
        class_labels = labels[class_indices]

        class_images, class_labels = shuffle(class_images, class_labels, random_state=42)

        split_idx = int(len(class_images) * (1 - test_size))

        X_train.extend(class_images[:split_idx])
        y_train.extend(class_labels[:split_idx])
        X_val.extend(class_images[split_idx:])
        y_val.extend(class_labels[split_idx:])

    return shuffle(np.array(X_train), np.array(y_train), random_state=42), \
           shuffle(np.array(X_val), np.array(y_val), random_state=42)


print("Cargando datos...")
images, labels, label_encoder = load_images()
print(f"🔹 Total imágenes cargadas: {len(images)}")
print(f"🔹 Clases detectadas: {dict(zip(label_encoder.classes_, Counter(labels)))}")

# División 80/20 por clase
(X_train, y_train), (X_val, y_val) = split_data_by_class(images, labels)

print("\n🔍 Datos después de dividir:")
print(f"🔸 Entrenamiento: {Counter(y_train)}")
print(f"🔸 Validación: {Counter(y_val)}")

# Crear y entrenar modelo
model = create_model()
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# Guardar modelo
model.save('src/lung_cancer_model.keras')
print("\n✅ Modelo entrenado y guardado con éxito.")


print("\nEvaluando modelo...")

y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)

# Clasificación
print("\n📊 Reporte de clasificación:")
print(classification_report(y_val, y_pred_classes, target_names=label_encoder.classes_))

# Matriz de confusión
cm = confusion_matrix(y_val, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_,
            yticklabels=label_encoder.classes_)
plt.title('Matriz de Confusión')
plt.xlabel('Predicción')
plt.ylabel('Realidad')
plt.show()

# Gráficas de entrenamiento
plt.figure(figsize=(12, 6))

# Precisión
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Entrenamiento')
plt.plot(history.history['val_accuracy'], label='Validación')
plt.title('Precisión del modelo')
plt.xlabel('Épocas')
plt.ylabel('Precisión')
plt.legend()

# Pérdida
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Entrenamiento')
plt.plot(history.history['val_loss'], label='Validación')
plt.title('Pérdida del modelo')
plt.xlabel('Épocas')
plt.ylabel('Pérdida')
plt.legend()

plt.tight_layout()
plt.show()
