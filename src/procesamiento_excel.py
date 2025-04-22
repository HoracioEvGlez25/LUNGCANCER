import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def cargar_archivos_excel():
    # Ruta de los archivos Excel
    excel_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'excel')
    
    # Verificar si los archivos existen
    print("Verificando si los archivos existen...")
    print(f"Ruta absoluta de la carpeta 'data/excel': {excel_dir}")
    
    # Listar archivos en el directorio
    archivos = [f for f in os.listdir(excel_dir) if f.endswith('.csv')]
    
    if len(archivos) == 0:
        print("No se encontraron archivos CSV en la carpeta.")
        return None, None
    
    # Cargar los archivos CSV en pandas
    print("Archivos encontrados. Cargando...")
    dataset1 = pd.read_csv(os.path.join(excel_dir, archivos[0]))
    dataset2 = pd.read_csv(os.path.join(excel_dir, archivos[1]))
    
    # Mostrar primeras filas de ambos datasets
    print(f"\nPrimeras filas de {archivos[0]}:")
    print(dataset1.head())
    print(f"\nPrimeras filas de {archivos[1]}:")
    print(dataset2.head())
    
    return dataset1, dataset2

def analisis_exploratorio(dataset1, dataset2):
    print("\nRealizando análisis exploratorio...")

    # Mostrar estadísticas descriptivas de ambos datasets
    print("\nEstadísticas descriptivas del dataset1:")
    print(dataset1.describe())
    
    print("\nEstadísticas descriptivas del dataset2:")
    print(dataset2.describe())

    # Filtrar solo las columnas numéricas
    dataset1_numeric = dataset1.select_dtypes(include=['float64', 'int64'])
    dataset2_numeric = dataset2.select_dtypes(include=['float64', 'int64'])

    # Mostrar la matriz de correlación de dataset1
    print("\nMatriz de correlación de dataset1:")
    plt.figure(figsize=(10, 8))
    sns.heatmap(dataset1_numeric.corr(), annot=True, cmap='coolwarm')
    plt.title("Matriz de Correlación - Dataset 1")
    plt.show()

    # Mostrar la matriz de correlación de dataset2
    print("\nMatriz de correlación de dataset2:")
    plt.figure(figsize=(10, 8))
    sns.heatmap(dataset2_numeric.corr(), annot=True, cmap='coolwarm')
    plt.title("Matriz de Correlación - Dataset 2")
    plt.show()

    # Verificar si hay valores nulos en ambos datasets
    print("\nValores nulos en dataset1:")
    print(dataset1.isnull().sum())

    print("\nValores nulos en dataset2:")
    print(dataset2.isnull().sum())

    # Mostrar las primeras filas de ambos datasets
    print("\nPrimeras filas de dataset1:")
    print(dataset1.head())

    print("\nPrimeras filas de dataset2:")
    print(dataset2.head())

# Cargar los datasets
dataset1, dataset2 = cargar_archivos_excel()

if dataset1 is not None and dataset2 is not None:
    # Realizar el análisis exploratorio si los datasets se cargaron correctamente
    analisis_exploratorio(dataset1, dataset2)
else:
    print("Error al cargar los datasets.")
