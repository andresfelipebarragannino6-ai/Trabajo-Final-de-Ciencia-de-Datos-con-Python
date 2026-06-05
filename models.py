# Importa la librería pandas para manipulación de datos
import pandas as pd

# Importa NumPy para operaciones numéricas
import numpy as np

# Importa Pipeline para encadenar preprocesamiento y modelo
from sklearn.pipeline import Pipeline

# Importa el modelo de Regresión Logística
from sklearn.linear_model import LogisticRegression

# Importa modelos basados en árboles de decisión
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

# Importa métricas de evaluación para clasificación
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)


# ==========================================================
# Entrenamiento de Modelos
# ==========================================================

# Función que entrena varios modelos de clasificación
def train_models(
    preprocessor,
    X_train,
    y_train
):
    """
    Entrena los modelos predictivos.
    """

    # Diccionario que almacena los modelos
    models = {

        # Modelo de Regresión Logística
        "Logistic Regression": Pipeline([

            # Paso de preprocesamiento
            ("preprocessor", preprocessor),

            # Paso del modelo
            (
                "model",
                LogisticRegression(
                    max_iter=2000,      # Máximo número de iteraciones
                    random_state=42,    # Semilla para reproducibilidad
                    solver="lbfgs"      # Algoritmo de optimización
                )
            )
        ]),

        # Modelo Random Forest
        "Random Forest": Pipeline([

            # Preprocesamiento
            ("preprocessor", preprocessor),

            # Clasificador Random Forest
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,      # Número de árboles
                    max_depth=10,          # Profundidad máxima
                    min_samples_split=5,   # Mínimo de muestras para dividir nodo
                    min_samples_leaf=2,    # Mínimo de muestras por hoja
                    random_state=42,       # Semilla
                    n_jobs=-1              # Usa todos los núcleos disponibles
                )
            )
        ]),

        # Modelo Gradient Boosting
        "Gradient Boosting": Pipeline([

            # Preprocesamiento
            ("preprocessor", preprocessor),

            # Clasificador Gradient Boosting
            (
                "model",
                GradientBoostingClassifier(
                    n_estimators=200,    # Número de árboles secuenciales
                    learning_rate=0.05, # Tasa de aprendizaje
                    max_depth=4,         # Profundidad máxima
                    random_state=42      # Semilla
                )
            )
        ])
    }

    # Recorre cada modelo del diccionario
    for model_name, model in models.items():

        # Entrena el pipeline completo
        model.fit(
            X_train,
            y_train
        )

    # Retorna los modelos entrenados
    return models


# ==========================================================
# Evaluación de Modelos
# ==========================================================

# Función que evalúa todos los modelos
def evaluate_models(
    models_dict,
    X_test,
    y_test
):
    """
    Devuelve métricas comparativas.
    """

    # Lista donde se almacenarán los resultados
    results = []

    # Recorre cada modelo
    for model_name, model in models_dict.items():

        # Genera predicciones
        y_pred = model.predict(X_test)

        try:

            # Obtiene probabilidades de la clase positiva
            y_prob = model.predict_proba(X_test)[:, 1]

            # Calcula el ROC-AUC
            roc_auc = roc_auc_score(
                y_test,
                y_prob
            )

        except:

            # Si el modelo no genera probabilidades
            roc_auc = np.nan

        # Guarda métricas calculadas
        results.append({

            # Nombre del modelo
            "Modelo": model_name,

            # Exactitud
            "Accuracy": round(
                accuracy_score(
                    y_test,
                    y_pred
                ),
                4
            ),

            # Precisión
            "Precision": round(
                precision_score(
                    y_test,
                    y_pred
                ),
                4
            ),

            # Sensibilidad (Recall)
            "Recall": round(
                recall_score(
                    y_test,
                    y_pred
                ),
                4
            ),

            # F1 Score
            "F1-Score": round(
                f1_score(
                    y_test,
                    y_pred
                ),
                4
            ),

            # Área bajo la curva ROC
            "ROC-AUC": round(
                roc_auc,
                4
            )
        })

    # Convierte los resultados en DataFrame
    results_df = pd.DataFrame(results)

    # Ordena por Recall descendente
    results_df = results_df.sort_values(
        by="Recall",
        ascending=False
    )

    # Reinicia índices
    results_df.reset_index(
        drop=True,
        inplace=True
    )

    # Devuelve tabla de resultados
    return results_df


# ==========================================================
# Mejor Modelo
# ==========================================================

# Función para obtener el mejor modelo
def get_best_model(
    models_dict,
    metrics_df
):
    """
    Retorna el mejor modelo según Recall.
    """

    # Obtiene el nombre del modelo con mayor Recall
    best_model_name = metrics_df.iloc[0]["Modelo"]

    # Recupera el modelo correspondiente
    best_model = models_dict[
        best_model_name
    ]

    # Devuelve nombre y modelo
    return (
        best_model_name,
        best_model
    )


# ==========================================================
# Matriz de Confusión
# ==========================================================

# Función que genera la matriz de confusión
def get_confusion_matrix(
    model,
    X_test,
    y_test
):
    """
    Calcula la matriz de confusión.
    """

    # Realiza predicciones
    y_pred = model.predict(
        X_test
    )

    # Calcula la matriz de confusión
    cm = confusion_matrix(
        y_test,
        y_pred
    )

    # Retorna la matriz
    return cm


# ==========================================================
# Reporte de Clasificación
# ==========================================================

# Función que genera un reporte completo de métricas
def get_classification_report(
    model,
    X_test,
    y_test
):
    """
    Devuelve métricas detalladas.
    """

    # Predicciones del modelo
    y_pred = model.predict(
        X_test
    )

    # Genera el reporte en formato diccionario
    report = classification_report(
        y_test,
        y_pred,
        output_dict=True
    )

    # Convierte el reporte a DataFrame
    return pd.DataFrame(
        report
    ).transpose()


# ==========================================================
# Importancia de Variables
# ==========================================================

# Función para identificar variables más relevantes
def get_feature_importance(
    best_model,
    feature_names,
    top_n=20
):
    """
    Obtiene las variables más importantes.
    """

    # Obtiene el estimador dentro del pipeline
    estimator = best_model.named_steps[
        "model"
    ]

    # Para modelos basados en árboles
    if hasattr(
        estimator,
        "feature_importances_"
    ):

        # Extrae importancia de variables
        importance = (
            estimator.feature_importances_
        )

    # Para modelos lineales
    elif hasattr(
        estimator,
        "coef_"
    ):

        # Usa el valor absoluto de los coeficientes
        importance = np.abs(
            estimator.coef_[0]
        )

    else:

        # Si el modelo no soporta importancia
        return pd.DataFrame()

    # Construye DataFrame de importancia
    importance_df = pd.DataFrame({

        "Feature": feature_names,
        "Importance": importance

    })

    # Ordena de mayor a menor importancia
    importance_df = (
        importance_df
        .sort_values(
            by="Importance",
            ascending=False
        )
        .head(top_n)
    )

    # Reinicia índices
    importance_df.reset_index(
        drop=True,
        inplace=True
    )

    # Devuelve el resultado
    return importance_df


# ==========================================================
# Predicción Individual
# ==========================================================

# Función para estimar riesgo de abandono de un cliente
def predict_customer_risk(
    model,
    customer_data
):
    """
    Predicción para el simulador.
    """

    # Predicción de clase (0 o 1)
    prediction = model.predict(
        customer_data
    )[0]

    # Probabilidad de abandono
    probability = (
        model.predict_proba(
            customer_data
        )[0][1]
    )

    # Devuelve clase y probabilidad
    return prediction, probability


# ==========================================================
# Curva ROC
# ==========================================================

# Función para obtener datos necesarios para graficar ROC
def get_roc_data(
    model,
    X_test,
    y_test
):
    """
    Devuelve datos ROC.
    """

    # Importa roc_curve localmente
    from sklearn.metrics import roc_curve

    # Obtiene probabilidades de clase positiva
    probabilities = (
        model.predict_proba(
            X_test
        )[:, 1]
    )

    # Calcula FPR, TPR y umbrales
    fpr, tpr, thresholds = roc_curve(
        y_test,
        probabilities
    )

    # Calcula área bajo la curva
    auc_score = roc_auc_score(
        y_test,
        probabilities
    )

    # Devuelve los valores necesarios para la gráfica
    return (
        fpr,
        tpr,
        auc_score
    )


# ==========================================================
# Segmentación de Riesgo
# ==========================================================

# Función para clasificar el nivel de riesgo
def classify_risk_level(
    probability
):
    """
    Clasifica el riesgo para el dashboard.
    """

    # Riesgo alto
    if probability >= 0.40:
        return "ALTO"

    # Riesgo medio
    elif probability >= 0.20:
        return "MEDIO"

    # Riesgo bajo
    else:
        return "BAJO"


# ==========================================================
# Resumen Ejecutivo
# ==========================================================

# Función que genera recomendaciones automáticas
def generate_business_summary(
    probability
):
    """
    Recomendación automática.
    """

    # Caso de riesgo alto
    if probability >= 0.40:

        return """
        Cliente con probabilidad crítica de abandono.

        Recomendaciones:

        • Contacto inmediato del equipo comercial.
        • Oferta personalizada de retención.
        • Revisión de incidentes recientes.
        • Incentivos económicos temporales.
        """

    # Caso de riesgo medio
    elif probability >= 0.20:

        return """
        Cliente con riesgo moderado.

        Recomendaciones:

        • Seguimiento preventivo.
        • Encuesta de satisfacción.
        • Beneficios de permanencia.
        """

    # Caso de riesgo bajo
    else:

        return """
        Cliente estable.

        Recomendaciones:

        • Mantener fidelización.
        • Cross-selling.
        • Upselling de servicios.
        """