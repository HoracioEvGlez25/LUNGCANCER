from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Cargar el modelo entrenado
model = load_model('lung_cancer_model.h5')

def preprocess_image(img_path):
    # Cargar la imagen, redimensionarla y preprocesarla
    img = image.load_img(img_path, target_size=(224, 224))  # Redimensiona a 224x224 (como en el modelo)
    img_array = image.img_to_array(img)  # Convertir la imagen a un array numpy
    img_array = np.expand_dims(img_array, axis=0)  # Expandir las dimensiones para que sea compatible con el modelo
    img_array = img_array / 255.0  # Normalizar la imagen
    return img_array

def predict_image(img_path):
    img_array = preprocess_image(img_path)
    prediction = model.predict(img_array)
    class_idx = np.argmax(prediction, axis=1)  # La clase con mayor probabilidad
    class_names = ['benigno', 'maligno', 'normal']  # Nombres de las clases
    predicted_class = class_names[class_idx[0]]
    print(f"La predicción para la imagen es: {predicted_class}")  # Mostrar la predicción

# Ruta de la imagen que deseas predecir
img_path = 'Benigno.jpg'  # Cambia esta ruta a la imagen que quieres predecir

# Realizar la predicción
predict_image(img_path)
