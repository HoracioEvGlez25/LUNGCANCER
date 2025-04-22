from flask import Flask, render_template, request, redirect, url_for, flash, session
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import joblib

app = Flask(__name__)
app.secret_key = 'tu_clave_secreta_aqui'
app.config['SESSION_TYPE'] = 'filesystem'

# Configuración para la subida de archivos
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Cargar el modelo de imágenes
MODEL_PATH = 'lung_cancer_model.h5'

if os.path.exists(MODEL_PATH):
    try:
        image_model = load_model(MODEL_PATH)
        print("✅ Modelo de imágenes cargado correctamente.")
    except Exception as e:
        print(f"❌ Error al cargar el modelo de imágenes: {e}")
        image_model = None
else:
    print(f"❌ No se encontró el archivo del modelo de imágenes: {MODEL_PATH}")
    image_model = None

# Cargar el modelo clínico
MODEL_CLINICO_PATH = 'lung_cancer_model_final.pkl'
try:
    modelo_completo = joblib.load(MODEL_CLINICO_PATH)
    if isinstance(modelo_completo, dict) and 'modelo' in modelo_completo:
        modelo_clinico = modelo_completo['modelo']
        umbral = modelo_completo.get('umbral', 0.5)
        print("✅ Modelo clínico cargado correctamente.")
    else:
        modelo_clinico = modelo_completo
        umbral = 0.5
        print("⚠ Modelo clínico cargado pero no tiene la estructura esperada.")
except Exception as e:
    print(f"❌ Error al cargar el modelo clínico: {e}")
    modelo_clinico = None
    umbral = 0.5

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def procesar_formulario(form):
    print("✅ Procesando datos del formulario clínico...")
    
    # Mapeo de datos del formulario a valores numéricos
    datos = {
 'SMOKING': 1 if form.get('smoking') == 'yes' else 2,
'YELLOW_FINGERS': 1 if form.get('yellow_fingers') == 'yes' else 2,
'ANXIETY': 1 if form.get('anxiety') == 'yes' else 2,
'PEER_PRESSURE': 1 if form.get('peer_pressure') == 'yes' else 2,
'CHRONIC_DISEASE': 1 if form.get('chronic_disease') == 'yes' else 2,
'FATIGUE': 1 if form.get('fatigue') == 'yes' else 2,
'ALLERGY': 1 if form.get('allergy') == 'yes' else 2,
'WHEEZING': 1 if form.get('wheezing') == 'yes' else 2,
'ALCOHOL_CONSUMING': 1 if form.get('alcohol_consuming') == 'yes' else 2,
'COUGHING': 1 if form.get('coughing') == 'yes' else 2,
'SHORTNESS_OF_BREATH': 1 if form.get('shortness_of_breath') == 'yes' else 2,
'SWALLOWING_DIFFICULTY': 1 if form.get('swallowing_difficulty') == 'yes' else 2,
'CHEST_PAIN': 1 if form.get('chest_pain') == 'yes' else 2

    }
    
    # Ordenar las características según lo que espera el modelo
    column_order = [
        'AGE', 'SMOKING', 'WHEEZING', 'ALCOHOL_CONSUMING', 'GENDER', 'PEER_PRESSURE', 
        'ALLERGY', 'SWALLOWING_DIFFICULTY', 'ANXIETY', 'SHORTNESS_OF_BREATH', 
        'YELLOW_FINGERS', 'COUGHING', 'FATIGUE', 'CHEST_PAIN', 'CHRONIC_DISEASE'
    ]
    
    # Asegurarse de que los nombres coincidan (corregir posibles diferencias)
    datos_corregidos = {}
    for col in column_order:
        # Buscar la clave en los datos sin importar mayúsculas/minúsculas o guiones bajos
        key_match = next((k for k in datos.keys() if k.upper().replace('_', '') == col.upper().replace('_', '')), None)
        if key_match:
            datos_corregidos[col] = datos[key_match]
        else:
            print(f"⚠ Advertencia: No se encontró la columna {col} en los datos")
            datos_corregidos[col] = 1  # Valor por defecto
    
    input_data = np.array([list(datos_corregidos.values())])
    
    if modelo_clinico is not None:
        try:
            # Obtener probabilidades
            proba = modelo_clinico.predict_proba(input_data)[:, 1]
            riesgo = float(proba[0])
            
            # Aplicar umbral
            diagnostico = "Positivo" if riesgo > umbral else "Negativo"
            detalles = f"Probabilidad: {riesgo*100:.2f}% (Umbral: {umbral*100:.1f}%)"
            
            session['diagnostico'] = diagnostico
            session['detalles'] = detalles
            session['tipo_prediccion'] = 'datos_clinicos'
            
            print(f"✅ Predicción realizada: {diagnostico} ({detalles})")
        except Exception as e:
            print(f"❌ Error al hacer predicción: {e}")
            flash("Error al procesar los datos clínicos. Por favor intente nuevamente.")
    else:
        flash("El modelo clínico no está disponible. Contacte al administrador.")

def procesar_imagen(file):
    if image_model is None:
        raise ValueError("Modelo de imágenes no disponible.")
    
    if file.filename == '':
        raise ValueError("No se seleccionó ningún archivo.")
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img_array = preprocess_image(filepath)
        prediction = image_model.predict(img_array)
        class_idx = np.argmax(prediction, axis=1)
        class_names = ['Benigno', 'Maligno', 'Normal']
        predicted_class = class_names[class_idx[0]]

        diagnostico = "Positivo" if predicted_class == 'Maligno' else "Negativo"
        detalles = f"Resultado del análisis: {predicted_class}"

        session['diagnostico'] = diagnostico
        session['detalles'] = detalles
        session['tipo_prediccion'] = 'imagen'
        session['imagen_path'] = filename
    else:
        raise ValueError("Archivo no válido. Suba una imagen JPG o PNG.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/equipo')
def equipo():
    return render_template('equipo.html')

@app.route('/estadisticas')
def estadisticas():
    return render_template('estadisticas.html')
@app.route('/modelos')
def modelos():
    return render_template('modelos.html')

@app.route('/prediccion', methods=['GET', 'POST'])
def prediccion():
    if request.method == 'POST':
        try:
            prediction_type = request.form.get('prediction_type')

            if prediction_type == 'form':
                procesar_formulario(request.form)
            elif prediction_type == 'image':
                file = request.files['radiografia']
                procesar_imagen(file)
            else:
                flash("Tipo de predicción no válido.")
                return redirect(request.url)

            return redirect(url_for('resultado'))

        except Exception as e:
            flash(f"Error en la predicción: {str(e)}")
            return redirect(request.url)

    return render_template('prediccion.html')

@app.route('/resultado')
def resultado():
    diagnostico = session.get('diagnostico')
    detalles = session.get('detalles')
    tipo_prediccion = session.get('tipo_prediccion')
    imagen_path = session.get('imagen_path', None)

    if not diagnostico or not detalles:
        flash("No hay resultados para mostrar. Realice una predicción primero.")
        return redirect(url_for('prediccion'))

    return render_template('result.html',
                         diagnostico=diagnostico,
                         detalles=detalles,
                         tipo_prediccion=tipo_prediccion,
                         imagen_path=imagen_path)

if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)