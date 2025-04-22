import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Ruta de las imágenes
IMAGE_DIR = "C:/Users/juane/OneDrive/Desktop/cancer_lung_prediction/data/imagenes/"
IMG_SIZE = (224, 224)  # Redimensionamos a 224x224
BATCH_SIZE = 32

def load_images():
    datagen = ImageDataGenerator(
        rescale=1.0/255,       # Normaliza los píxeles entre 0 y 1
        validation_split=0.2,  # 80% entrenamiento, 20% validación
        rotation_range=20,     # Augmentación: rotaciones
        zoom_range=0.2,        
        horizontal_flip=True   
    )

    # Carga las imágenes desde el directorio
    train_data = datagen.flow_from_directory(
        IMAGE_DIR,            # Carpeta con las imágenes
        target_size=IMG_SIZE, # Redimensionar las imágenes
        batch_size=BATCH_SIZE,
        class_mode='categorical',  # Como tenemos más de dos clases, usamos 'categorical'
        subset="training"     # Usamos el 80% para entrenamiento
    )

    # Carga las imágenes para validación
    val_data = datagen.flow_from_directory(
        IMAGE_DIR,            # Carpeta con las imágenes
        target_size=IMG_SIZE, # Redimensionar las imágenes
        batch_size=BATCH_SIZE,
        class_mode='categorical',  # Como tenemos más de dos clases, usamos 'categorical'
        subset="validation"   # Usamos el 20% para validación
    )

    print("\n✅ Datos cargados correctamente!")
    print(f"🔹 Imágenes de entrenamiento: {train_data.samples}")
    print(f"🔹 Imágenes de validación: {val_data.samples}")
    print(f"📂 Clases detectadas: {train_data.class_indices}")

    return train_data, val_data

# Ejecutar el script
if __name__ == "__main__":
    train_data, val_data = load_images()
