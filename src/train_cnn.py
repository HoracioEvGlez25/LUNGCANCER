from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from src.preprocesing import cargar_imagenes  # Usamos la función de preprocesamiento

def crear_modelo():
    print("Creando el modelo...")
    model = Sequential()

    # Capa de convolución
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Segunda capa de convolución
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Capa de aplanado
    model.add(Flatten())

    # Capa densa
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))  # Dropout para prevenir sobreajuste
    model.add(Dense(3, activation='softmax'))  # 3 clases (benigno, maligno, normal)

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    print("Modelo creado.")
    return model

# Cargar los datos
print("Cargando datos...")
train_data, val_data = cargar_imagenes()  # Llamamos a la función de preprocesamiento

print("Datos cargados. Entrenando el modelo...")

# Crear el modelo
model = crear_modelo()

# Entrenar el modelo
model.fit(train_data, epochs=10, validation_data=val_data)

# Guardar el modelo entrenado
model.save('src/lung_cancer_model.h5')

print("\n✅ El modelo ha sido entrenado y guardado exitosamente!")

# Evaluar el modelo con los datos de validación
print("\nEvaluando el modelo en datos de validación...")

# Predecir las etiquetas de las imágenes de validación
y_pred = model.predict(val_data)
y_true = val_data.classes  # Las clases verdaderas

# Mostrar el reporte de clasificación
print("\nReporte de clasificación:")
print(classification_report(y_true, np.argmax(y_pred, axis=1)))

# Mostrar la matriz de confusión
print("\nMatriz de confusión:")
print(confusion_matrix(y_true, np.argmax(y_pred, axis=1)))
