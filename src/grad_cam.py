import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

# Cargar el modelo previamente entrenado
model = load_model('src/lung_cancer_model.h5')  # O 'lung_cancer_model.keras'

# Función para preprocesar la imagen
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Redimensionar la imagen
    img_array = image.img_to_array(img) / 255.0  # Normalizar
    img_array = np.expand_dims(img_array, axis=0)  # Cambiar la forma a (1, 224, 224, 3)
    return img_array, img

# Función para generar el mapa Grad-CAM
def generate_grad_cam(model, img_array, img, last_conv_layer_name="conv2d_2", pred_index=None):
    # Obtener la salida de la última capa convolucional
    last_conv_layer = model.get_layer(last_conv_layer_name)
    last_conv_layer_model = tf.keras.models.Model(model.input, last_conv_layer.output)

    # Obtener la salida de las predicciones (logits)
    classifier_layer_model = tf.keras.models.Model(model.input, model.output)

    # Calcular el gradiente de la clase objetivo con respecto a la salida de la última capa convolucional
    with tf.GradientTape() as tape:
        last_conv_layer_output = last_conv_layer_model(img_array)
        tape.watch(last_conv_layer_output)
        preds = classifier_layer_model(img_array)
        if not pred_index:
            pred_index = np.argmax(preds[0])  # Si no se especifica un índice, tomamos el índice de la clase con mayor probabilidad
    
    # Obtener el gradiente de la clase objetivo con respecto a la última capa convolucional
    grads = tape.gradient(preds[:, pred_index], last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))  # Promedio de los gradientes

    # Multiplicar los gradientes por las activaciones de la última capa convolucional
    last_conv_layer_output = last_conv_layer_output[0]
    pooled_grads = pooled_grads.numpy()
    for i in range(last_conv_layer_output.shape[-1]):
        last_conv_layer_output[:, :, i] *= pooled_grads[i]

    # Crear el mapa de calor
    heatmap = np.mean(last_conv_layer_output, axis=-1)  # Promediar las activaciones a lo largo del canal de características
    heatmap = np.maximum(heatmap, 0)  # Eliminar los valores negativos
    heatmap = cv2.resize(heatmap, (img.size[0], img.size[1]))  # Redimensionar el mapa de calor al tamaño original de la imagen
    heatmap /= np.max(heatmap)  # Normalizar el mapa de calor

    return heatmap

# Función para mostrar el mapa Grad-CAM sobre la imagen
def display_grad_cam(img_path, heatmap):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = np.uint8(255 * heatmap)  # Escalar el mapa de calor entre 0 y 255
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)  # Aplicar el mapa de color JET

    # Superponer el mapa de calor sobre la imagen original
    superimposed_img = heatmap * 0.4 + img  # Mezclar la imagen y el mapa de calor (ajustar el factor de opacidad)
    superimposed_img = np.uint8(np.clip(superimposed_img, 0, 255))  # Asegurarse de que los valores estén entre 0 y 255

    # Mostrar la imagen con el mapa Grad-CAM
    plt.imshow(superimposed_img)
    plt.axis('off')  # Desactivar los ejes
    plt.show()

# Ruta de la imagen de entrada (reemplaza por la ruta correcta o la imagen que suba el usuario)
img_path = 'uploads/imagen_ejemplo.jpg'  # Asegúrate de que esta sea la ruta correcta

# Preprocesar la imagen
img_array, img = preprocess_image(img_path)

# Generar el mapa Grad-CAM
heatmap = generate_grad_cam(model, img_array, img)

# Mostrar el mapa Grad-CAM sobre la imagen original
display_grad_cam(img_path, heatmap)
