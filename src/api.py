from flask import Flask, request, jsonify
from src.model import entrenar_modelo_imagenes  # Suponiendo que tienes una función para el entrenamiento de imágenes
from utils.procesamiento_excel import entrenar_modelo_tabular  # Suponiendo que tienes la función de entrenamiento para datos tabulares
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
import numpy as np

app = Flask(__name__)

@app.route('/entrenar', methods=['POST'])
def entrenar_modelo_completo():
    """
    Endpoint para entrenar el modelo con ambos tipos de datos (imágenes y datos tabulares).
    """
    # Cargar los datos de imágenes y datos tabulares
    try:
        # Cargar y preprocesar imágenes
        train_data_imagenes, val_data_imagenes = cargar_imagenes()  # Esta función debe estar definida en el archivo adecuado

        # Cargar y preprocesar los datos tabulares
        dataset1, dataset2 = cargar_archivos_excel()  # Esta función debe cargar los datos tabulares
        tabular_data = preprocesar_datos_tabulares(dataset1, dataset2)  # Preprocesamos los datos

        # Dividir los datos tabulares en entrenamiento y validación
        tabular_data_train, tabular_data_val = train_test_split(tabular_data, test_size=0.2, random_state=42)

        # Entrenar el modelo de imágenes
        model_imagenes = entrenar_modelo_imagenes(train_data_imagenes, val_data_imagenes, epochs=10, batch_size=32)

        # Entrenar el modelo de datos tabulares
        model_tabular = entrenar_modelo_tabular(tabular_data_train, tabular_data_val, epochs=10, batch_size=32)

        # Guardar ambos modelos
        model_imagenes.save('src/lung_cancer_model_imagenes.h5')
        model_tabular.save('src/lung_cancer_model_tabular.h5')

        return jsonify({"message": "Modelos entrenados y guardados exitosamente!"}), 200

    except Exception as e:
        return jsonify({"message": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
