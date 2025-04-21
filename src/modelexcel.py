import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, f1_score, roc_curve, roc_auc_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import matplotlib.pyplot as plt

def cargar_y_combinar_datasets():
    # Cargar los dos datasets
    dataset1 = pd.read_csv('C:/Users/juane/OneDrive/Desktop/cancer_lung_prediction/data/excel/dataset1.csv')
    dataset2 = pd.read_csv('C:/Users/juane/OneDrive/Desktop/cancer_lung_prediction/data/excel/dataset2.csv')

    # Limpiar nombres de columnas
    for df in [dataset1, dataset2]:
        df.columns = df.columns.str.strip().str.replace(' ', '_')

    # Encontrar columnas comunes (sin contar la columna 'LUNG_CANCER')
    common_cols = list(set(dataset1.columns) & set(dataset2.columns))
    if 'LUNG_CANCER' in common_cols:
        common_cols.remove('LUNG_CANCER')

    # Combinar los datasets
    combined = pd.concat([dataset1[common_cols + ['LUNG_CANCER']],
                          dataset2[common_cols + ['LUNG_CANCER']]], ignore_index=True)

    print("\nDistribución de clases en los datasets combinados:")
    print(combined['LUNG_CANCER'].value_counts())

    # Feedback de las variables
    print("\nVariables utilizadas en el modelo:")
    print(combined.columns.tolist())

    # Cantidad de variables y registros
    print(f"\nCantidad de variables (columnas): {len(combined.columns)}")
    print(f"Cantidad de registros (filas): {len(combined)}")

    return combined

def preprocesar_datos(df):
    # Mapeo de la variable objetivo 'LUNG_CANCER'
    df['LUNG_CANCER'] = df['LUNG_CANCER'].map({'YES': 1, 'NO': 0, '1': 1, '0': 0, 1: 1, 0: 0}).fillna(0)

    # Columnas binarias (para preprocesar como 1 y 2)
    binary_cols = [
        'GENDER', 'SMOKING', 'YELLOW_FINGERS', 'ANXIETY', 'PEER_PRESSURE', 'CHRONIC_DISEASE',
        'FATIGUE', 'ALLERGY', 'WHEEZING', 'ALCOHOL_CONSUMING', 'COUGHING', 'SHORTNESS_OF_BREATH',
        'SWALLOWING_DIFFICULTY', 'CHEST_PAIN'
    ]

    print("\nPreprocesando las variables binarias y otras columnas...")

    # Mostrar ejemplos de valores para algunas columnas
    for col in binary_cols:
        if col in df.columns:
            print(f"\nColumna {col}:")
            print(f"Valores únicos antes de procesar: {df[col].unique()}")
            df[col] = df[col].astype(str).str.upper().map({
                'YES': 2, 'NO': 1, '2': 2, '1': 1, '1.0': 1, '2.0': 2
            }).fillna(1)
            print(f"Valores únicos después de procesar: {df[col].unique()}")

    # Imputación de datos faltantes
    imputer = SimpleImputer(strategy='most_frequent')
    df_imputed = imputer.fit_transform(df)
    df = pd.DataFrame(df_imputed, columns=df.columns)

    # Asegurarse de que las columnas binarias son de tipo entero
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    df['AGE'] = df['AGE'].astype(int)
    df['LUNG_CANCER'] = df['LUNG_CANCER'].astype(int)

    print("\nPreprocesamiento completo. Datos listos para entrenar el modelo.")
    print(f"Cantidad de registros después del preprocesamiento: {len(df)}")
    print(f"Cantidad de variables después del preprocesamiento: {df.shape[1]}")

    return df.drop('LUNG_CANCER', axis=1), df['LUNG_CANCER']

def entrenar_y_guardar_modelo():
    print("Cargando datasets...")
    df = cargar_y_combinar_datasets()

    print("Preprocesando datos...")
    X, y = preprocesar_datos(df)

    # Mostrar detalles de las primeras filas de X y y
    print("\nMuestra de los datos de entrada (X) y objetivo (y) después del preprocesamiento:")
    print(X.head())
    print(y.head())

    print("\nBalance inicial de clases:")
    print(y.value_counts())

    print("\nAplicando SMOTE para balanceo de clases...")
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    print("Balance después de SMOTE:")
    print(pd.Series(y_res).value_counts())

    print("\nEntrenando modelo con XGBoost...")
    modelo = XGBClassifier(
    n_estimators=500,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=1,
    scale_pos_weight=(y == 0).sum() / (y == 1).sum(),
    random_state=42,
    use_label_encoder=False,
    eval_metric='logloss'
)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    y_probs = cross_val_predict(modelo, X_res, y_res, cv=skf, method='predict_proba')[:, 1]
    y_test_full = y_res

    print("\nCalculando umbral óptimo según F1...")
    thresholds = np.linspace(0.1, 0.9, 80)
    f1_scores = [f1_score(y_test_full, (y_probs > t).astype(int), pos_label=1) for t in thresholds]
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    print(f"Umbral óptimo encontrado: {optimal_threshold:.3f}")

    print("\nEvaluación con umbral óptimo:")
    y_pred_opt = (y_probs > optimal_threshold).astype(int)
    print(classification_report(y_test_full, y_pred_opt, target_names=['No Cáncer', 'Cáncer']))

    print("\nAUC ROC:", roc_auc_score(y_test_full, y_probs))

    fpr, tpr, _ = roc_curve(y_test_full, y_probs)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label='Curva ROC')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title("Curva ROC")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nEntrenando modelo final con todo el set balanceado...")
    modelo.fit(X_res, y_res)

    print("\nGuardando modelo y umbral en archivo .pkl...")
    modelo_completo = {
        'modelo': modelo,
        'umbral': optimal_threshold
    }
    joblib.dump(modelo_completo, 'lung_cancer_model_final.pkl')
    print("✅ Modelo guardado correctamente como 'lung_cancer_model_final.pkl'")

if __name__ == '__main__':
    entrenar_y_guardar_modelo()
