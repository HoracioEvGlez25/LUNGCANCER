import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

def limpiar_datos(dataset):
    print("🔍 Limpiando datos...")
    dataset = dataset.dropna()  # Eliminar filas con valores nulos
    dataset = dataset.drop_duplicates()  # Eliminar duplicados
    print(f"✅ Datos limpios. Filas restantes: {len(dataset)}")
    return dataset

def codificar_etiquetas(dataset):
    print("🔠 Codificando etiquetas...")
    if 'Etiqueta' in dataset.columns:
        encoder = LabelEncoder()
        dataset['Etiqueta'] = encoder.fit_transform(dataset['Etiqueta'])
        print("✅ Etiquetas codificadas correctamente.")
    else:
        print("⚠️ Advertencia: La columna 'Etiqueta' no se encuentra en el dataset.")
    return dataset

def normalizar_datos(dataset, metodo='standard'):
    print(f"📊 Normalizando datos usando el método: {metodo}...")
    columnas_numericas = dataset.select_dtypes(include=['float64', 'int64']).columns
    
    if len(columnas_numericas) == 0:
        print("⚠️ Advertencia: No hay columnas numéricas para normalizar.")
        return dataset
    
    scaler = StandardScaler() if metodo == 'standard' else MinMaxScaler()
    dataset[columnas_numericas] = scaler.fit_transform(dataset[columnas_numericas])
    print("✅ Datos normalizados correctamente.")
    return dataset

def dividir_datos(dataset, test_size=0.2):
    print(f"✂️ Dividiendo datos en entrenamiento ({1-test_size:.0%}) y prueba ({test_size:.0%})...")
    if 'Etiqueta' not in dataset.columns:
        raise KeyError("Error: La columna 'Etiqueta' no está presente en el dataset.")
    
    X = dataset.drop(columns=['Etiqueta'])
    y = dataset['Etiqueta']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    print(f"✅ División completada. Tamaño de entrenamiento: {len(X_train)}, prueba: {len(X_test)}")
    return X_train, X_test, y_train, y_test

def cargar_archivos_excel():
    print("📂 Cargando archivos Excel...")
    try:
        dataset1 = pd.read_csv(r'C:/Users/juane/OneDrive/Desktop/cancer_lung_prediction/data/excel/dataset1.csv')
        dataset2 = pd.read_csv(r'C:/Users/juane/OneDrive/Desktop/cancer_lung_prediction/data/excel/dataset2.csv')
        print("✅ Archivos cargados exitosamente.")
        return dataset1, dataset2
    except FileNotFoundError as e:
        print(f"❌ Error: No se encontró el archivo. {e}")
        exit()

if __name__ == "__main__":
    print("🚀 Iniciando procesamiento de datos...")
    dataset1, dataset2 = cargar_archivos_excel()
    
    # Verificar los nombres exactos de las columnas (sin espacios ni caracteres extraños)
    print("Nombres exactos de las columnas en dataset1:", dataset1.columns.tolist())
    print("Nombres exactos de las columnas en dataset2:", dataset2.columns.tolist())
    
    # Limpiar los nombres de las columnas, eliminando espacios
    dataset1.columns = dataset1.columns.str.strip()
    dataset2.columns = dataset2.columns.str.strip()

    # Renombrar la columna 'LUNG_CANCER' a 'Etiqueta'
    if 'LUNG_CANCER' in dataset1.columns:
        print("⚠️ Renombrando 'LUNG_CANCER' a 'Etiqueta' en dataset1...")
        dataset1.rename(columns={'LUNG_CANCER': 'Etiqueta'}, inplace=True)
    
    if 'LUNG_CANCER' in dataset2.columns:
        print("⚠️ Renombrando 'LUNG_CANCER' a 'Etiqueta' en dataset2...")
        dataset2.rename(columns={'LUNG_CANCER': 'Etiqueta'}, inplace=True)
    
    # Limpiar y procesar los datos
    dataset1 = limpiar_datos(dataset1)
    dataset2 = limpiar_datos(dataset2)
    
    dataset1 = codificar_etiquetas(dataset1)
    dataset2 = codificar_etiquetas(dataset2)
    
    dataset1 = normalizar_datos(dataset1, metodo='standard')
    dataset2 = normalizar_datos(dataset2, metodo='minmax')
    
    if 'Etiqueta' in dataset1.columns:
        X_train, X_test, y_train, y_test = dividir_datos(dataset1)
    else:
        print("❌ Error: No se pudo dividir dataset1 porque no tiene la columna 'Etiqueta'.")
    
    print("🎯 Procesamiento finalizado con éxito.")
