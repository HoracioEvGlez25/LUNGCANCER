import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing import image
from tensorflow.keras import models, layers
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report

# Función para cargar y preprocesar imágenes
def cargar_imagenes():
    print("Cargando imágenes...")
    image_folder = 'C:/Users/juane/OneDrive/Desktop/cancer_lung_prediction/data/imagenes'
    
    benigno_files = os.listdir(os.path.join(image_folder, 'benigno'))
    maligno_files = os.listdir(os.path.join(image_folder, 'maligno'))
    normal_files = os.listdir(os.path.join(image_folder, 'normal'))

    images, labels = [], []
    categorias = {'benigno': 0, 'maligno': 1, 'normal': 2}

    for categoria, label in categorias.items():
        folder_path = os.path.join(image_folder, categoria)
        for img in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img)
            img_array = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img_array) / 255.0
            images.append(img_array)
            labels.append(label)

    images = np.array(images)
    labels = to_categorical(np.array(labels), num_classes=3)
    print(f"Total imágenes cargadas: {len(images)}")
    return images, labels

# Crear modelo CNN
def crear_modelo_imagenes():
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(3, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Función para cargar y preprocesar datos desde archivos de Excel
def cargar_datos_excel():
    print("Cargando datos de Excel...")
    excel_file1 = 'data/excel/dataset1.csv'
    excel_file2 = 'data/excel/dataset2.csv'
    
    dataset = pd.concat([pd.read_csv(excel_file1), pd.read_csv(excel_file2)])
    dataset['LUNG_CANCER'] = dataset['LUNG_CANCER'].map({'YES': 1, 'NO': 0})
    dataset['GENDER'] = LabelEncoder().fit_transform(dataset['GENDER'])
    dataset = dataset.drop(columns=['PEER_PRESSURE', 'SWALLOWING_DIFFICULTY', 'CHEST_PAIN'])
    
    X = StandardScaler().fit_transform(dataset.drop(columns=['LUNG_CANCER']))
    y = to_categorical(dataset['LUNG_CANCER'], num_classes=2)
    
    print(f"Total registros cargados: {len(X)}")
    return X, y

# Crear modelo para datos tabulares
def crear_modelo_excel(input_shape):
    model = models.Sequential([
        layers.Dense(64, activation='relu', input_shape=(input_shape,)),
        layers.Dropout(0.5),
        layers.Dense(2, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Función para graficar los resultados del entrenamiento
def graficar_historial(historial, titulo):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(historial.history['accuracy'], label='Precisión entrenamiento')
    plt.plot(historial.history['val_accuracy'], label='Precisión validación')
    plt.xlabel('Épocas')
    plt.ylabel('Precisión')
    plt.title(f'{titulo} - Precisión')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(historial.history['loss'], label='Pérdida entrenamiento')
    plt.plot(historial.history['val_loss'], label='Pérdida validación')
    plt.xlabel('Épocas')
    plt.ylabel('Pérdida')
    plt.title(f'{titulo} - Pérdida')
    plt.legend()
    
    plt.show()

# Función para evaluar el modelo con una matriz de confusión
def evaluar_modelo(modelo, X_test, y_test, nombre_modelo):
    y_pred = np.argmax(modelo.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Sí'], yticklabels=['No', 'Sí'])
    plt.xlabel('Predicción')
    plt.ylabel('Real')
    plt.title(f'Matriz de Confusión - {nombre_modelo}')
    plt.show()
    
    print(f'Reporte de Clasificación - {nombre_modelo}\n', classification_report(y_true, y_pred))

# Función principal
def main():
    images, labels = cargar_imagenes()
    X_excel, y_excel = cargar_datos_excel()
    
    X_train_images, X_test_images, y_train_images, y_test_images = train_test_split(images, labels, test_size=0.2, random_state=42)
    X_train_excel, X_test_excel, y_train_excel, y_test_excel = train_test_split(X_excel, y_excel, test_size=0.2, random_state=42)

    print("Entrenando modelo de imágenes...")
    modelo_imagenes = crear_modelo_imagenes()
    historial_imagenes = modelo_imagenes.fit(X_train_images, y_train_images, epochs=5, batch_size=32, validation_data=(X_test_images, y_test_images))
    graficar_historial(historial_imagenes, "Modelo de Imágenes")
    evaluar_modelo(modelo_imagenes, X_test_images, y_test_images, "Modelo de Imágenes")

    print("Entrenando modelo de Excel...")
    modelo_excel = crear_modelo_excel(X_train_excel.shape[1])
    historial_excel = modelo_excel.fit(X_train_excel, y_train_excel, epochs=5, batch_size=32, validation_data=(X_test_excel, y_test_excel))
    graficar_historial(historial_excel, "Modelo de Datos Excel")
    evaluar_modelo(modelo_excel, X_test_excel, y_test_excel, "Modelo de Datos Excel")

    modelo_imagenes.save('lung_cancer_model_imagenesintegrado.h5')
    modelo_excel.save('lung_cancer_model_excelintegrado.h5')
    print("Modelos guardados correctamente.")

if __name__ == "__main__":
    main()
