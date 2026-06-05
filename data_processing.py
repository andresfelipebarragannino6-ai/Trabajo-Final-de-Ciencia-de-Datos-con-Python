# Importa la librería pandas para manipulación y análisis de datos
import pandas as pd

# Importa la librería NumPy para operaciones numéricas
import numpy as np

# Importa ColumnTransformer para aplicar diferentes transformaciones a distintos tipos de columnas
from sklearn.compose import ColumnTransformer

# Importa StandardScaler para normalizar variables numéricas
# e OneHotEncoder para codificar variables categóricas
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Importa la función para dividir los datos en entrenamiento y prueba
from sklearn.model_selection import train_test_split


# ==========================================================
# Limpieza de Outliers
# ==========================================================

# Función que elimina valores atípicos utilizando el método IQR
def remove_outliers_iqr(df, columns):

    """
    Elimina outliers utilizando el método IQR.
    """

    # Crea una copia del DataFrame para no modificar el original
    df_clean = df.copy()

    # Recorre cada columna especificada
    for col in columns:

        # Calcula el primer cuartil (25%)
        q1 = df_clean[col].quantile(0.25)

        # Calcula el tercer cuartil (75%)
        q3 = df_clean[col].quantile(0.75)

        # Calcula el rango intercuartílico
        iqr = q3 - q1

        # Define el límite inferior aceptable
        lower_bound = q1 - (1.5 * iqr)

        # Define el límite superior aceptable
        upper_bound = q3 + (1.5 * iqr)

        # Conserva únicamente los registros dentro de los límites
        df_clean = df_clean[
            (df_clean[col] >= lower_bound)
            &
            (df_clean[col] <= upper_bound)
        ]

    # Retorna el DataFrame sin outliers
    return df_clean


# ==========================================================
# Carga y Limpieza de Datos
# ==========================================================

# Función para cargar y limpiar el dataset
def load_and_clean_data(file_path):

    """
    Carga y prepara el dataset Telco Customer Churn.
    """

    # Lee el archivo CSV y lo convierte en DataFrame
    df = pd.read_csv(file_path)

    # ------------------------------------------------------
    # Eliminar CustomerID
    # ------------------------------------------------------

    # Verifica si existe la columna customerID
    if "customerID" in df.columns:

        # Elimina la columna porque no aporta valor predictivo
        df.drop(columns=["customerID"], inplace=True)

    # ------------------------------------------------------
    # Limpiar TotalCharges
    # ------------------------------------------------------

    # Reemplaza espacios vacíos por valores nulos (NaN)
    df["TotalCharges"] = (
        df["TotalCharges"]
        .replace(" ", np.nan)
    )

    # Convierte la columna a valores numéricos
    # Los errores se convierten en NaN
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Reemplaza los valores faltantes con la mediana
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    # ------------------------------------------------------
    # Eliminar duplicados
    # ------------------------------------------------------

    # Elimina filas duplicadas del DataFrame
    df.drop_duplicates(inplace=True)

    # ------------------------------------------------------
    # Convertir columnas binarias
    # ------------------------------------------------------

    # Lista de columnas binarias
    binary_columns = [
        "Partner",
        "Dependents",
        "PhoneService",
        "PaperlessBilling",
        "Churn"
    ]

    # Recorre cada columna binaria
    for col in binary_columns:

        # Verifica que la columna exista
        if col in df.columns:

            # Normaliza los valores Yes y No
            df[col] = df[col].replace({
                "Yes": "Yes",
                "No": "No"
            })

    # ------------------------------------------------------
    # Eliminar outliers
    # ------------------------------------------------------

    # Lista de variables numéricas donde se buscarán outliers
    numeric_columns = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges"
    ]

    # Elimina los valores atípicos utilizando IQR
    df = remove_outliers_iqr(
        df,
        numeric_columns
    )

    # ------------------------------------------------------
    # Reset index
    # ------------------------------------------------------

    # Reinicia el índice después de eliminar filas
    df.reset_index(
        drop=True,
        inplace=True
    )

    # Devuelve el DataFrame limpio
    return df


# ==========================================================
# Preprocesamiento
# ==========================================================

# Función que construye el pipeline de transformación
def get_preprocessing_pipeline(df):

    """
    Construye el pipeline de transformación.
    """

    # Crea una copia del DataFrame
    df_copy = df.copy()

    # Convierte la variable objetivo Churn a formato numérico
    df_copy["Churn"] = df_copy["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    # Separa las variables predictoras
    X = df_copy.drop(
        columns=["Churn"]
    )

    # Selecciona las variables numéricas
    numeric_features = (
        X.select_dtypes(
            include=[
                "int64",
                "float64"
            ]
        )
        .columns
        .tolist()
    )

    # Selecciona las variables categóricas
    categorical_features = (
        X.select_dtypes(
            include=["object"]
        )
        .columns
        .tolist()
    )

    # Construye el transformador de columnas
    preprocessor = ColumnTransformer(
        transformers=[

            # Escala las variables numéricas
            (
                "num",
                StandardScaler(),
                numeric_features
            ),

            # Codifica las variables categóricas
            (
                "cat",
                OneHotEncoder(
                    handle_unknown="ignore",  # Ignora categorías nuevas
                    sparse_output=False       # Devuelve matriz densa
                ),
                categorical_features
            )
        ],

        # Descarta columnas no especificadas
        remainder="drop"
    )

    # Devuelve el preprocesador configurado
    return preprocessor


# ==========================================================
# División Train Test
# ==========================================================

# Función que divide los datos en entrenamiento y prueba
def prepare_train_test_split(df):

    """
    Divide el dataset en entrenamiento y prueba.
    """

    # Crea una copia del DataFrame
    dataset = df.copy()

    # Convierte la variable objetivo a formato binario
    dataset["Churn"] = dataset["Churn"].map({
        "Yes": 1,
        "No": 0
    })

    # Variables predictoras
    X = dataset.drop(
        columns=["Churn"]
    )

    # Variable objetivo
    y = dataset["Churn"]

    # Divide el dataset en entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(

        X,              # Variables de entrada
        y,              # Variable objetivo
        test_size=0.20, # 20% para prueba
        stratify=y,     # Mantiene proporciones de clases
        random_state=42 # Garantiza reproducibilidad
    )

    # Retorna los conjuntos generados
    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


# ==========================================================
# Feature Names
# ==========================================================

# Función para obtener los nombres de las variables transformadas
def get_feature_names(preprocessor):

    """
    Obtiene nombres de variables transformadas.
    """

    try:
        # Obtiene los nombres generados por el preprocesador
        return preprocessor.get_feature_names_out()

    except Exception:
        # Si ocurre algún error devuelve una lista vacía
        return []