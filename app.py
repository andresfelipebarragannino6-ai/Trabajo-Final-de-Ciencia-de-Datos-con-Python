import streamlit as st
from groq import Groq

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak
)

from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter

from io import BytesIO
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.metrics import (
    confusion_matrix
)

from data_processing import (
    load_and_clean_data,
    get_preprocessing_pipeline,
    prepare_train_test_split,
    get_feature_names
)

from models import (
    train_models,
    evaluate_models,
    get_best_model,
    get_feature_importance,
    get_confusion_matrix,
    get_roc_data,
    predict_customer_risk,
    classify_risk_level,
    generate_business_summary
)

import os

from groq import Groq

from dotenv import load_dotenv

load_dotenv()

import os
from groq import Groq

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

client = Groq(
    api_key=GROQ_API_KEY
)

# ==========================================================
# CONFIGURACIÓN DE PÁGINA
# ==========================================================

st.set_page_config(
    page_title="Telco Customer Churn Dashboard",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================================
# SESSION STATE
# ==========================================================

if "last_simulation" not in st.session_state:
    st.session_state.last_simulation = None

if "confirm_report" not in st.session_state:
    st.session_state.confirm_report = False

# ==========================================================
# ESTILOS CSS
# ==========================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size:40px;
        font-weight:bold;
        color:#1f77b4;
    }

    .subtitle {
        font-size:18px;
        color:gray;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# ==========================================================
# HEADER
# ==========================================================

st.markdown(
    """
    <div class='main-title'>
    📉 Dashboard Inteligente de Churn Telecom
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class='subtitle'>
    Plataforma Analítica para Predicción y Retención de Clientes
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# ==========================================================
# SIDEBAR
# ==========================================================

with st.sidebar:

    st.image(
        "https://cdn-icons-png.flaticon.com/512/3050/3050525.png",
        width=120
    )

    st.header("⚙ Configuración")

    st.success(
        """
        Sistema de apoyo a la toma de decisiones
        para estrategias de retención.
        """
    )

    st.markdown("---")

    st.markdown(
        """
        ### Objetivos del Dashboard

        - Comprender el comportamiento de fuga.
        - Analizar factores de riesgo.
        - Comparar modelos predictivos.
        - Simular clientes en tiempo real.
        """
    )

# ==========================================================
# CARGA DE DATOS
# ==========================================================

@st.cache_data
def load_dataset():

    df = load_and_clean_data(
        "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    return df


df = load_dataset()

# ==========================================================
# ENTRENAMIENTO
# ==========================================================

@st.cache_resource
def train_pipeline(df):

    X_train, X_test, y_train, y_test = (
        prepare_train_test_split(df)
    )

    preprocessor = (
        get_preprocessing_pipeline(df)
    )

    models_dict = train_models(
        preprocessor,
        X_train,
        y_train
    )

    metrics_df = evaluate_models(
        models_dict,
        X_test,
        y_test
    )

    best_model_name, best_model = (
        get_best_model(
            models_dict,
            metrics_df
        )
    )

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        models_dict,
        metrics_df,
        best_model_name,
        best_model
    )

(
    X_train,
    X_test,
    y_train,
    y_test,
    preprocessor,
    models_dict,
    metrics_df,
    best_model_name,
    best_model
) = train_pipeline(df)

# ==========================================================
# VARIABLES IMPORTANTES
# ==========================================================

feature_names = (
    best_model
    .named_steps["preprocessor"]
    .get_feature_names_out()
)

importance_df = get_feature_importance(
    best_model,
    feature_names,
    top_n=15
)

# ==========================================================
# KPIs
# ==========================================================

total_customers = len(df)

churn_rate = (
    (df["Churn"] == "Yes")
    .mean() * 100
)

avg_monthly = (
    df["MonthlyCharges"]
    .mean()
)

avg_tenure = (
    df["tenure"]
    .mean()
)

# ==========================================================
# TABS
# ==========================================================

tab1, tab2, tab3 = st.tabs(
    [
        "📊 Análisis Estratégico de Fuga",
        "🤖 Diagnóstico del Modelo Predictivo",
        "🎯 Simulador de Riesgo en Vivo"
    ]
)
# ==========================================================
# TAB 1
# ANÁLISIS ESTRATÉGICO DE FUGA
# ==========================================================

with tab1:

    st.info(
        """
        📌 **Contexto Estratégico**

        La pérdida de clientes (Churn) representa uno de los mayores riesgos
        financieros para las empresas de telecomunicaciones.

        Diversos estudios indican que adquirir un nuevo cliente puede costar
        entre 5 y 7 veces más que retener uno existente.

        Por esta razón, comprender qué factores impulsan la cancelación de
        servicios permite diseñar estrategias de retención más efectivas y
        maximizar el valor de vida del cliente (Customer Lifetime Value).
        """
    )

    # ======================================================
    # KPIs
    # ======================================================

    st.subheader("📈 Indicadores Clave del Negocio")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Clientes Totales",
            f"{total_customers:,}"
        )

    with col2:
        st.metric(
            "Tasa de Churn",
            f"{churn_rate:.2f}%"
        )

    with col3:
        st.metric(
            "Cargo Mensual Promedio",
            f"${avg_monthly:.2f}"
        )

    with col4:
        st.metric(
            "Antigüedad Promedio",
            f"{avg_tenure:.1f} meses"
        )

    st.markdown("---")

    # ======================================================
    # DISTRIBUCIÓN DE CHURN
    # ======================================================

    st.subheader("1️⃣ Distribución General de Clientes")

    churn_counts = (
        df["Churn"]
        .value_counts()
        .reset_index()
    )

    churn_counts.columns = [
        "Estado",
        "Cantidad"
    ]

    fig_pie = px.pie(
        churn_counts,
        names="Estado",
        values="Cantidad",
        hole=0.4,
        title="Distribución General del Churn"
    )

    st.plotly_chart(
        fig_pie,
        use_container_width=True
    )

    with st.expander(
        "💡 Insight y Acción de Negocio"
    ):

        st.markdown(
            """
### ¿Qué estamos observando?

Este gráfico muestra la proporción total de clientes que permanecen en la empresa frente a aquellos que han cancelado sus servicios.

### ¿Por qué es importante?

Cada cliente perdido representa una disminución directa en los ingresos recurrentes de la compañía.

Además, recuperar un cliente perdido o adquirir uno nuevo suele ser considerablemente más costoso que mantener uno existente.

### Interpretación

Una tasa elevada de churn puede indicar problemas relacionados con:

- Calidad del servicio.
- Atención al cliente.
- Competencia agresiva.
- Estrategias de precios.
- Experiencia del usuario.

### Acción recomendada

Implementar sistemas de alerta temprana que permitan identificar clientes en riesgo antes de que tomen la decisión de abandonar.
            """
        )

    st.markdown("---")

    # ======================================================
    # CONTRATO
    # ======================================================

    st.subheader("2️⃣ Impacto del Tipo de Contrato")

    contract_churn = pd.crosstab(
        df["Contract"],
        df["Churn"],
        normalize="index"
    ) * 100

    contract_churn = (
        contract_churn
        .reset_index()
        .melt(
            id_vars="Contract",
            var_name="Churn",
            value_name="Porcentaje"
        )
    )

    fig_contract = px.bar(
        contract_churn,
        x="Contract",
        y="Porcentaje",
        color="Churn",
        barmode="group",
        title="Churn según Tipo de Contrato"
    )

    st.plotly_chart(
        fig_contract,
        use_container_width=True
    )

    with st.expander(
        "💡 Insight y Acción de Negocio"
    ):

        st.markdown(
            """
### ¿Qué estamos observando?

Se compara la tasa de abandono entre contratos mensuales, anuales y de dos años.

### Hallazgo esperado

Los clientes con contratos mes a mes suelen presentar los niveles más altos de abandono.

### ¿Por qué ocurre?

Estos clientes poseen una barrera de salida muy baja y pueden cancelar el servicio con facilidad ante cualquier inconveniente.

### Impacto empresarial

Los contratos largos generan ingresos más estables y reducen significativamente la volatilidad del negocio.

### Acción recomendada

- Incentivar migraciones hacia contratos anuales.
- Ofrecer descuentos por permanencia.
- Diseñar beneficios exclusivos para contratos de largo plazo.
            """
        )

    st.markdown("---")

    # ======================================================
    # TENURE
    # ======================================================

    st.subheader("3️⃣ Lealtad del Cliente")

    fig_tenure = px.box(
        df,
        x="Churn",
        y="tenure",
        color="Churn",
        title="Antigüedad de Clientes vs Churn"
    )

    st.plotly_chart(
        fig_tenure,
        use_container_width=True
    )

    with st.expander(
        "💡 Insight y Acción de Negocio"
    ):

        st.markdown(
            """
### ¿Qué estamos observando?

Se analiza la antigüedad de los clientes que permanecen y de aquellos que abandonan.

### Hallazgo esperado

Las cancelaciones suelen concentrarse durante los primeros meses de relación con la empresa.

### ¿Por qué ocurre?

Los clientes nuevos aún están evaluando si el servicio cumple sus expectativas.

Problemas durante el onboarding pueden acelerar la decisión de abandono.

### Acción recomendada

Implementar programas de acompañamiento durante los primeros 6 a 12 meses.

### Beneficio esperado

La reducción de la fuga temprana suele generar uno de los mayores retornos sobre inversión en estrategias de retención.
            """
        )

    st.markdown("---")

    # ======================================================
    # INTERNET SERVICE
    # ======================================================

    st.subheader("4️⃣ Riesgo de Abandono por Servicio de Internet")

    internet_churn = pd.crosstab(
        df["InternetService"],
        df["Churn"],
        normalize="index"
    ) * 100

    internet_churn = (
        internet_churn
        .reset_index()
        .melt(
            id_vars="InternetService",
            var_name="Churn",
            value_name="Porcentaje"
        )
    )

    fig_internet = px.bar(
        internet_churn,
        x="InternetService",
        y="Porcentaje",
        color="Churn",
        barmode="group",
        title="Churn por Tipo de Internet"
    )

    st.plotly_chart(
        fig_internet,
        use_container_width=True
    )

    with st.expander(
        "💡 Insight y Acción de Negocio"
    ):

        st.markdown(
            """
### ¿Qué estamos observando?

Analizamos cómo varía la tasa de abandono según el servicio de internet contratado.

### Posibles causas

Si un segmento presenta más churn, puede deberse a:

- Problemas técnicos.
- Mala experiencia de usuario.
- Precios elevados.
- Competencia más agresiva.

### Acción recomendada

Investigar los segmentos con mayores tasas de abandono y diseñar campañas específicas para ellos.

### Beneficio esperado

Reducir el churn en estos grupos suele tener un impacto directo sobre los ingresos recurrentes.
            """
        )

    st.markdown("---")

    # ======================================================
    # RELACIÓN FINANCIERA
    # ======================================================

    st.subheader("5️⃣ Relación Financiera del Cliente")

    fig_finance = px.scatter(
        df,
        x="MonthlyCharges",
        y="TotalCharges",
        color="Churn",
        opacity=0.7,
        title="Monthly Charges vs Total Charges"
    )

    st.plotly_chart(
        fig_finance,
        use_container_width=True
    )

    with st.expander(
        "💡 Insight y Acción de Negocio"
    ):

        st.markdown(
            """
### ¿Qué estamos observando?

Este gráfico relaciona el valor mensual pagado por cada cliente con su gasto acumulado histórico.

### Interpretación

Permite identificar si los clientes con mayores cargos mensuales tienen una mayor tendencia al abandono.

### ¿Por qué ocurre?

Los clientes premium suelen tener expectativas más altas respecto al servicio recibido.

Si perciben que el valor entregado no justifica el costo, aumenta significativamente el riesgo de cancelación.

### Acción recomendada

- Crear programas VIP de fidelización.
- Identificar clientes de alto valor.
- Implementar beneficios personalizados.

### Impacto esperado

Retener clientes de alto valor económico genera un efecto financiero significativamente mayor para la compañía.
            """
        )

    st.markdown("---")

    st.success(
        """
        ✅ Conclusión Ejecutiva

        Los factores relacionados con permanencia, tipo de contrato,
        características del servicio y cargos económicos suelen ser
        los principales impulsores del abandono de clientes.

        Comprender estas relaciones permite diseñar estrategias
        de retención más eficientes y rentables.
        """
    )
    # ==========================================================
    # TAB 2
    # DIAGNÓSTICO DEL MODELO PREDICTIVO
    # ==========================================================

    with tab2:

        st.header("🤖 Diagnóstico del Modelo Predictivo")

        st.info(
            """
            Esta sección permite evaluar el rendimiento de los algoritmos de Machine Learning
            utilizados para predecir el abandono de clientes.

            El objetivo principal no es únicamente lograr una alta precisión,
            sino identificar oportunamente a los clientes con riesgo real de fuga.
            """
        )

        # ======================================================
        # COMPARACIÓN DE MODELOS
        # ======================================================

        st.subheader("📊 Comparación de Modelos")

        metrics_display = metrics_df.copy()

        metric_columns = [
            "Accuracy",
            "Precision",
            "Recall",
            "F1-Score",
            "ROC-AUC"
        ]

        for col in metric_columns:
            metrics_display[col] = (
                metrics_display[col] * 100
            ).round(2)

        st.dataframe(
            metrics_display,
            use_container_width=True,
            hide_index=True
        )

        st.success(
            f"""
            🏆 Mejor modelo seleccionado:

            **{best_model_name}**

            El criterio de selección utilizado es el Recall,
            ya que minimizar la pérdida de clientes es la prioridad estratégica.
            """
        )

        # ======================================================
        # RECALL
        # ======================================================

        st.subheader("🎯 ¿Por qué el Recall es tan importante?")

        st.warning(
            """
            Recall responde a la siguiente pregunta:

            ¿De todos los clientes que realmente abandonarán la compañía,
            cuántos fue capaz de detectar el modelo?

            En estrategias de retención es preferible contactar algunos clientes
            adicionales que perder clientes valiosos por no haberlos identificado.

            Por ello, un Recall elevado suele ser más importante que maximizar
            únicamente la precisión del modelo.
            """
        )

        st.markdown("---")

        # ======================================================
        # MATRIZ DE CONFUSIÓN
        # ======================================================

        st.subheader("📌 Matriz de Confusión")

        cm = get_confusion_matrix(
            best_model,
            X_test,
            y_test
        )

        fig_cm = px.imshow(
            cm,
            text_auto=True,
            aspect="auto",
            labels=dict(
                x="Predicción",
                y="Valor Real",
                color="Cantidad"
            ),
            x=[
                "Permanece",
                "Abandona"
            ],
            y=[
                "Permanece",
                "Abandona"
            ],
            title=f"Matriz de Confusión - {best_model_name}"
        )

        st.plotly_chart(
            fig_cm,
            use_container_width=True
        )

        with st.expander(
            "💡 Interpretación de Negocio"
        ):

            st.markdown(
                """
    ### Verdaderos Positivos (Permanece -> Permanece) 

    Son 933 clientes que realmente iban a abandonar y fueron identificados correctamente.

    ### Falsos Negativos (Abandona -> Abandona) 

    Son 195 clientes que abandonaron pero el modelo no detectó.

    Este es el error más costoso para la empresa.

    ### Verdaderos Negativos (Abandona -> Permanece)

    Son 117 clientes correctamente identificados como estables.

    ### Falsos Positivos (Permanece -> Abandona)

    Son 100 clientes marcados como riesgo cuando realmente no abandonarían.

    Aunque generan costos operativos, suelen ser menos perjudiciales
    que los falsos negativos.
                """
            )

        st.markdown("---")

        # ======================================================
        # CURVAS ROC
        # ======================================================

        st.subheader("📈 Curva ROC Comparativa")

        roc_fig = go.Figure()

        for model_name, model in models_dict.items():

            fpr, tpr, auc_score = get_roc_data(
                model,
                X_test,
                y_test
            )

            roc_fig.add_trace(

                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{model_name} (AUC={auc_score:.3f})"
                )

            )

        roc_fig.add_trace(

            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                name="Clasificador Aleatorio",
                line=dict(dash="dash")
            )

        )

        roc_fig.update_layout(
            title="Comparación de Curvas ROC",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate",
            height=600
        )

        st.plotly_chart(
            roc_fig,
            use_container_width=True
        )

        with st.expander(
            "💡 ¿Qué significa esta gráfica?"
        ):

            st.markdown(
                """
    La Curva ROC mide la capacidad del modelo para separar clientes
    que abandonarán de aquellos que permanecerán.

    ### Interpretación

    - AUC = 0.50 → Sin capacidad predictiva.
    - AUC > 0.70 → Buen modelo.
    - AUC > 0.80 → Modelo muy sólido.
    - AUC > 0.90 → Excelente discriminación.

    Cuanto más cercana esté la curva a la esquina superior izquierda,
    mejor será el desempeño del modelo.
                """
            )

        st.markdown("---")

        # ======================================================
        # IMPORTANCIA DE VARIABLES
        # ======================================================

        st.subheader("📌 Variables Más Influyentes")

        fig_importance = px.bar(
            importance_df.sort_values(
                by="Importance"
            ),
            x="Importance",
            y="Feature",
            orientation="h",
            title="Top Variables Predictivas"
        )

        st.plotly_chart(
            fig_importance,
            use_container_width=True
        )

        top_feature = (
            importance_df
            .sort_values(
                by="Importance",
                ascending=False
            )
            .iloc[0]["Feature"]
        )

        st.success(
            f"""
    ### Conclusión Ejecutiva

    La variable con mayor impacto en la predicción es:

    **{top_feature}**

    Esto significa que dicha característica tiene una fuerte influencia
    sobre la decisión de permanencia o abandono de los clientes.

    Estas variables representan las principales palancas de negocio
    sobre las que la organización debe actuar para reducir el churn.
            """
        )

        st.markdown("---")

        # ======================================================
        # TOP 10 VARIABLES
        # ======================================================

        st.subheader("🔍 Ranking de Factores de Riesgo")

        ranking_df = (
            importance_df
            .sort_values(
                by="Importance",
                ascending=False
            )
            .reset_index(drop=True)
        )

        ranking_df.index = ranking_df.index + 1

        st.dataframe(
            ranking_df,
            use_container_width=True
        )

        st.info(
            """
            Las variables mejor posicionadas suelen estar relacionadas con:

            • Tipo de contrato.
            • Antigüedad del cliente.
            • Cargos mensuales.
            • Servicios adicionales.
            • Soporte técnico.
            • Seguridad en línea.

            Estos factores constituyen las áreas prioritarias para las estrategias de retención.
            """
        )
# ==========================================================
# TAB 3
# SIMULADOR DE RIESGO
# ==========================================================

with tab3:

    st.header("🎯 Simulador de Riesgo de Fuga")

    st.info(
        """
        Esta herramienta permite simular clientes nuevos o existentes
        para estimar su probabilidad de abandono utilizando el mejor
        modelo de Machine Learning seleccionado.

        El objetivo es apoyar la toma de decisiones comerciales
        en tiempo real.
        """
    )

    # ======================================================
    # FORMULARIO
    # ======================================================

    with st.form("customer_simulation_form"):

        st.subheader("📋 Información del Cliente")

        col1, col2, col3 = st.columns(3)

        with col1:

            gender = st.selectbox(
                "Género",
                ["Male", "Female"]
            )

            senior = st.selectbox(
                "Adulto Mayor",
                [0, 1]
            )

            partner = st.selectbox(
                "Tiene Pareja",
                ["Yes", "No"]
            )

            dependents = st.selectbox(
                "Dependientes",
                ["Yes", "No"]
            )

        with col2:

            contract = st.selectbox(
                "Tipo de Contrato",
                [
                    "Month-to-month",
                    "One year",
                    "Two year"
                ]
            )

            paperless = st.selectbox(
                "Facturación Electrónica",
                ["Yes", "No"]
            )

            payment = st.selectbox(
                "Método de Pago",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)"
                ]
            )

        with col3:

            internet = st.selectbox(
                "Servicio de Internet",
                [
                    "DSL",
                    "Fiber optic",
                    "No"
                ]
            )

            online_security = st.selectbox(
                "Seguridad Online",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

            tech_support = st.selectbox(
                "Soporte Técnico",
                [
                    "Yes",
                    "No",
                    "No internet service"
                ]
            )

        st.markdown("---")

        st.subheader("💰 Información Comercial")

        col4, col5 = st.columns(2)

        with col4:

            tenure = st.slider(
                "Antigüedad (Meses)",
                min_value=0,
                max_value=72,
                value=12
            )

        with col5:

            monthly_charges = st.slider(
                "Cargo Mensual ($)",
                min_value=18.0,
                max_value=120.0,
                value=70.0
            )

        submit_button = st.form_submit_button(
            "🚨 Calcular Riesgo de Fuga"
        )

    # ======================================================
    # PREDICCIÓN
    # ======================================================

    if submit_button:

        total_charges = tenure * monthly_charges

        customer_data = pd.DataFrame([{

            "gender": gender,

            "SeniorCitizen": senior,

            "Partner": partner,

            "Dependents": dependents,

            "tenure": tenure,

            "PhoneService": "Yes",

            "MultipleLines": "No",

            "InternetService": internet,

            "OnlineSecurity": online_security,

            "OnlineBackup": "No",

            "DeviceProtection": "No",

            "TechSupport": tech_support,

            "StreamingTV": "No",

            "StreamingMovies": "No",

            "Contract": contract,

            "PaperlessBilling": paperless,

            "PaymentMethod": payment,

            "MonthlyCharges": monthly_charges,

            "TotalCharges": total_charges

        }])

        prediction, probability = predict_customer_risk(
            best_model,
            customer_data
        )

        # Convertir a tipos Python nativos
        prediction = int(prediction)
        probability = float(probability)

        # Clasificación personalizada
        if probability >= 0.40:
            risk_level = "ALTO"
        elif probability >= 0.20:
            risk_level = "MEDIO"
        else:
            risk_level = "BAJO"

        # Guardar simulación para el informe
        st.session_state.last_simulation = {
            "prediction": prediction,
            "probability": probability,
            "risk_level": risk_level,
            "data": customer_data.iloc[0].to_dict()
        }

        st.markdown("---")

        st.subheader("📊 Resultado de la Evaluación")

        # ==================================================
        # ALERTA PRINCIPAL
        # ==================================================

        if prediction == 1:

            st.error(
                "🚨 ALTO RIESGO DE FUGA"
            )

        else:

            st.success(
                "✅ CLIENTE SEGURO / BAJO RIESGO"
            )

        # ==================================================
        # MÉTRICAS
        # ==================================================

        col_a, col_b = st.columns(2)

        with col_a:

            st.metric(
                "Probabilidad de Abandono",
                f"{probability*100:.2f}%"
            )

        with col_b:

            st.metric(
                "Nivel de Riesgo",
                risk_level
            )

        st.progress(
            float(probability)
        )

        # ==================================================
        # INTERPRETACIÓN
        # ==================================================

        if probability >= 0.40:

            st.error(
                """
### 🔴 Riesgo Crítico

El cliente presenta una probabilidad extremadamente alta de cancelar el servicio.

Se recomienda intervención inmediata.
                """
            )

        elif probability >= 0.20:

            st.warning(
                """
### 🟠 Riesgo Moderado

El cliente presenta señales importantes de posible abandono.

Se recomienda seguimiento preventivo.
                """
            )

        else:

            st.success(
                """
### 🟢 Riesgo Bajo

Actualmente el cliente presenta alta probabilidad de permanencia.
                """
            )

        st.markdown("---")

        # ==================================================
        # RECOMENDACIONES AUTOMÁTICAS
        # ==================================================

        st.subheader("💡 Recomendación Comercial Automática")

        recommendation = generate_business_summary(
            probability
        )

        st.info(recommendation)

        # ==================================================
        # REGLAS DE NEGOCIO
        # ==================================================

        st.subheader("🎯 Plan de Acción Personalizado")

        recommendations = []

        if contract == "Month-to-month":

            recommendations.append(
                "Ofrecer migración a contrato anual con descuento."
            )

        if internet == "Fiber optic":

            recommendations.append(
                "Evaluar satisfacción del servicio de Fibra Óptica."
            )

        if tenure < 12:

            recommendations.append(
                "Aplicar programa de fidelización temprana."
            )

        if monthly_charges > 80:

            recommendations.append(
                "Revisar percepción de valor respecto al precio."
            )

        if online_security == "No":

            recommendations.append(
                "Promocionar servicios de seguridad online."
            )

        if tech_support == "No":

            recommendations.append(
                "Ofrecer plan preferencial de soporte técnico."
            )

        if len(recommendations) == 0:

            recommendations.append(
                "Mantener seguimiento comercial periódico."
            )

        for idx, rec in enumerate(
            recommendations,
            start=1
        ):

            st.write(
                f"{idx}. {rec}"
            )

        st.markdown("---")

        # ==================================================
        # RESULTADO DE LA EVALUACIÓN
        # ==================================================

        st.markdown("---")

        st.subheader("📊 Resultado de la Evaluación")

        # ==================================================
        # CLASIFICACIÓN DE RIESGO
        # ==================================================

        if probability >= 0.40:

            risk_level = "ALTO"

            st.error(
                f"""
        🚨 RIESGO ALTO DE ABANDONO

        Probabilidad estimada de fuga: {probability*100:.2f}%

        El cliente presenta señales muy fuertes de que abandonará la compañía en el corto plazo si no se realiza una intervención inmediata.
        """
            )

        elif probability >= 0.20:

            risk_level = "MEDIO"

            st.warning(
                f"""
        ⚠️ POSIBLE INTENCIÓN DE ABANDONO

        Probabilidad estimada de fuga: {probability*100:.2f}%

        El cliente muestra comportamientos asociados al abandono, pero aún no parece haber tomado una decisión definitiva.

        Este es el momento ideal para aplicar estrategias preventivas de retención.
        """
            )

        else:

            risk_level = "BAJO"

            st.success(
                f"""
        ✅ CLIENTE ESTABLE

        Probabilidad estimada de fuga: {probability*100:.2f}%

        Actualmente el cliente no presenta señales relevantes de abandono y tiene una alta probabilidad de continuar utilizando los servicios.
        """
            )

        # ==================================================
        # MÉTRICAS PRINCIPALES
        # ==================================================

        col_a, col_b = st.columns(2)

        with col_a:

            st.metric(
                "Probabilidad de Abandono",
                f"{probability*100:.2f}%"
            )

        with col_b:

            st.metric(
                "Nivel de Riesgo",
                risk_level
            )

        # ==================================================
        # INDICADOR VISUAL DE RIESGO
        # ==================================================

        st.subheader("📊 Semáforo de Riesgo")

        if probability >= 0.40:

            st.markdown(
                """
                <div style="
                    background-color:#dc3545;
                    padding:20px;
                    border-radius:10px;
                    color:white;
                    text-align:center;
                    font-size:24px;
                    font-weight:bold;">
                    🔴 RIESGO ALTO DE ABANDONO
                </div>
                """,
                unsafe_allow_html=True
            )

        elif probability >= 0.20:

            st.markdown(
                """
                <div style="
                    background-color:#ffc107;
                    padding:20px;
                    border-radius:10px;
                    color:black;
                    text-align:center;
                    font-size:24px;
                    font-weight:bold;">
                    🟡 POSIBLE INTENCIÓN DE ABANDONO
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            st.markdown(
                """
                <div style="
                    background-color:#198754;
                    padding:20px;
                    border-radius:10px;
                    color:white;
                    text-align:center;
                    font-size:24px;
                    font-weight:bold;">
                    🟢 CLIENTE ESTABLE
                </div>
                """,
                unsafe_allow_html=True
            )

        st.progress(float(probability))

        # ==================================================
        # INTERPRETACIÓN EJECUTIVA
        # ==================================================

        st.subheader("📈 Interpretación del Resultado")

        if probability >= 0.40:

            st.markdown(
                """
        ### 🔴 Riesgo Alto

        Este cliente tiene una probabilidad elevada de cancelar el servicio.

        #### Implicaciones para el negocio

        - Alta probabilidad de pérdida de ingresos.
        - Riesgo de migración hacia un competidor.
        - Posible insatisfacción con el servicio actual.

        #### Acción recomendada

        - Contacto inmediato por parte del equipo comercial.
        - Oferta personalizada de retención.
        - Revisión de incidencias recientes.
        - Incentivos económicos o beneficios exclusivos.
        """
            )

        elif probability >= 0.20:

            st.markdown(
                """
        ### 🟡 Riesgo Medio

        El cliente presenta señales tempranas asociadas al abandono.

        #### Implicaciones para el negocio

        - Existe una oportunidad importante de retención.
        - El cliente aún no parece haber tomado una decisión definitiva.

        #### Acción recomendada

        - Seguimiento preventivo.
        - Encuesta de satisfacción.
        - Oferta de beneficios adicionales.
        - Evaluar posibles inconformidades.
        """
            )

        else:

            st.markdown(
                """
        ### 🟢 Riesgo Bajo

        El cliente muestra una alta probabilidad de permanencia.

        #### Implicaciones para el negocio

        - Comportamiento estable.
        - Bajo riesgo de cancelación.

        #### Acción recomendada

        - Mantener estrategias de fidelización.
        - Ofrecer productos complementarios.
        - Programas de beneficios y recompensas.
        """
            )

        # ==================================================
        # RECOMENDACIÓN AUTOMÁTICA
        # ==================================================

        st.subheader("💡 Recomendación Comercial Automática")

        recommendation = generate_business_summary(
            probability
        )

        st.info(recommendation)

        # ==================================================
        # REGLAS DE NEGOCIO PERSONALIZADAS
        # ==================================================

        st.subheader("🎯 Plan de Acción Personalizado")

        recommendations = []

        if contract == "Month-to-month":

            recommendations.append(
                "Ofrecer migración a contrato anual con beneficios exclusivos."
            )

        if internet == "Fiber optic":

            recommendations.append(
                "Evaluar experiencia y satisfacción del servicio de fibra óptica."
            )

        if tenure < 12:

            recommendations.append(
                "Aplicar programa de fidelización para clientes nuevos."
            )

        if monthly_charges > 80:

            recommendations.append(
                "Revisar percepción de valor frente al costo mensual."
            )

        if online_security == "No":

            recommendations.append(
                "Promover servicios de seguridad online para aumentar vinculación."
            )

        if tech_support == "No":

            recommendations.append(
                "Ofrecer soporte técnico preferencial."
            )

        if len(recommendations) == 0:

            recommendations.append(
                "Mantener seguimiento comercial periódico."
            )

        for i, rec in enumerate(recommendations, start=1):

            st.write(f"**{i}.** {rec}")

        # ==================================================
        # RESUMEN EJECUTIVO
        # ==================================================

        if risk_level == "ALTO":

            interpretation = (
                "Cliente con alta probabilidad de abandono."
            )

        elif risk_level == "MEDIO":

            interpretation = (
                "Cliente con señales de posible abandono."
            )

        else:

            interpretation = (
                "Cliente estable sin intención aparente de abandonar."
            )

        st.success(
            f"""
        ### 📌 Resumen Ejecutivo

        **Modelo utilizado:** {best_model_name}

        **Probabilidad de abandono:** {probability*100:.2f}%

        **Clasificación:** {risk_level}

        **Interpretación:** {interpretation}

        **Antigüedad:** {tenure} meses

        **Cargo mensual:** ${monthly_charges:.2f}

        ### Recomendación Final

        Utilice esta clasificación para priorizar campañas de retención y asignar recursos comerciales de manera más eficiente.

        🔴 Alto → Intervención inmediata

        🟡 Medio → Seguimiento preventivo

        🟢 Bajo → Fidelización y crecimiento
        """
        )
        # Mostrar resultados de la simulación actual si existe en memoria
    if ("last_simulation" in st.session_state and st.session_state.last_simulation is not None):
        sim = st.session_state.last_simulation
        
        st.markdown("---")
        st.subheader("📊 Resultado de la Evaluación Realizada")
        
        if sim["prediction"] == 1:
            st.error(f"🚨 ALTO RIESGO DE FUGA (Nivel: {sim['risk_level']})")
        else:
            st.success(f"✅ CLIENTE SEGURO / BAJO RIESGO (Nivel: {sim['risk_level']})")

        col_a, col_b = st.columns(2)
        col_a.metric("Probabilidad de Abandono", f"{sim['probability']*100:.2f}%")
        col_b.metric("Nivel de Riesgo", sim["risk_level"])
        st.progress(sim["probability"])

        # ==========================================================
        # SECCIÓN DEL INFORME INTELIGENTE CON IA DE GROQ Y PDF
        # ==========================================================
        st.markdown("---")
        st.subheader("🤖 Generación de Informe Ejecutivo Inteligente (IA Groq)")
        
        st.warning(
            "⚠️ Antes de descargar el informe técnico para el cliente, confirme si desea basarlo "
            "en los datos de la última simulación mostrada arriba."
        )

        # Botonera de flujo de toma de decisiones requerida por el usuario
        c1, c2 = st.columns(2)
        with c1:
            if st.button("👍 Sí, estoy seguro de usar esta simulación"):
                st.session_state.confirm_report = True
        with c2:
            if st.button("👎 No, quiero realizar una nueva simulación"):
                st.session_state.last_simulation = None
                st.session_state.confirm_report = False
                st.rerun()

        # Si el usuario confirma que "Sí", se habilita el motor analítico de Groq y la descarga automática del PDF
        if ("confirm_report" in st.session_state and st.session_state.confirm_report):
            st.success("✨ Confirmación recibida de forma exitosa. Compilando análisis de Groq...")
            
            # Función helper para llamar de forma inteligente a la API de Groq pasándole el contexto dinámico
            def obtener_analisis_ia_groq(kpis, sim_data):
                prompt = f"""
                Actúa como un Director Analítico de Operaciones en Telecomunicaciones. Genera un informe ejecutivo condensado y persuasivo de alto nivel para presentar a la junta y al cliente basado en los siguientes datos reales recopilados:
                
                MÉTRICAS DEL NEGOCIO (DASHBOARD GENERAL):
                - Clientes Totales Analizados: {kpis['total_customers']}
                - Tasa de Deserción General (Churn Rate): {kpis['churn_rate']:.2f}%
                - Cargo Mensual Promedio General: ${kpis['avg_monthly']:.2f}
                - Antigüedad Promedio del Portafolio: {kpis['avg_tenure']:.1f} meses
                
                ÚLTIMA SIMULACIÓN DE CLIENTE EVALUADO EN TIEMPO REAL:
                - Tipo de Contrato: {sim_data['data']['Contract']}
                - Meses de Antigüedad: {sim_data['data']['tenure']}
                - Cargo Mensual Individual: ${sim_data['data']['MonthlyCharges']:.2f}
                - Servicio de Internet: {sim_data['data']['InternetService']}
                - Soporte Técnico Contratado: {sim_data['data']['TechSupport']}
                - Probabilidad de Fuga Estimada por el Modelo: {sim_data['probability']*100:.2f}%
                - Diagnóstico de Riesgo: {sim_data['risk_level']}
                
                Estructura tu respuesta exactamente con estas 3 secciones claras (usa lenguaje profesional en español):
                1. RESUMEN EJECUTIVO DE SITUACIÓN: Analiza el impacto de la tasa de churn general frente al caso simulado.
                2. DIAGNÓSTICO DEL CASO SIMULADO: Explica técnicamente por qué este cliente tiene ese riesgo según sus variables de contrato y cargos.
                3. RECOMENDACIÓN ESTRATÉGICA PARA LA TOMA DE DECISIONES: Define una oferta comercial viable para retener a este cliente de forma inmediata.
                """
                try:
                    chat_completion = client.chat.completions.create(
                        messages=[
                            {"role": "system", "content": "Eres un consultor estratégico experto en analítica de clientes y retención."},
                            {"role": "user", "content": prompt}
                        ],
                        model="llama-3.1-8b-instant",
                        temperature=0.3
                    )
                    return chat_completion.choices[0].message.content
                except Exception as e:
                    return f"Error procesando la solicitud en Groq: {str(e)}. No obstante, puede descargar los datos estructurados."

            # Ejecutar consulta a la IA
            kpis_negocio = {
                "total_customers": total_customers, "churn_rate": churn_rate,
                "avg_monthly": avg_monthly, "avg_tenure": avg_tenure
            }
            
            with st.spinner("IA de Groq analizando métricas financieras e individuales..."):
                texto_informe_groq = obtener_analisis_ia_groq(kpis_negocio, sim)
            
            # Desplegar el informe generado por la IA en pantalla
            st.markdown("### 📝 Informe Ejecutivo Inteligente de Groq")
            st.info(texto_informe_groq)

            # ==========================================================
            # MOTOR DE GENERACIÓN AUTOMÁTICA DE PDF (REPORTLAB)
            # ==========================================================
            def exportar_informe_pdf(texto_ia, kpis, sim_data):
                buffer = BytesIO()
                doc = SimpleDocTemplate(
                buffer,
                pagesize=letter,
                rightMargin=40,
                leftMargin=40,
                topMargin=40,
                bottomMargin=40
                )
                
                styles = getSampleStyleSheet()
                
                # Definición de estilos profesionales para ReportLab
                style_title = ParagraphStyle(
                    'TitleStyle', parent=styles['Heading1'],
                    fontSize=22, textColor=colors.HexColor("#1f77b4"),
                    spaceAfter=15, bold=True
                )
                style_subtitle = ParagraphStyle(
                    'SubStyle', parent=styles['Normal'],
                    fontSize=11, textColor=colors.gray, spaceAfter=20
                )
                style_h2 = ParagraphStyle(
                    'H2Style', parent=styles['Heading2'],
                    fontSize=14, textColor=colors.HexColor("#2c3e50"),
                    spaceBefore=12, spaceAfter=8, bold=True
                )
                style_body = ParagraphStyle(
                    'BodyStyle', parent=styles['Normal'],
                    fontSize=10.5, leading=14, textColor=colors.HexColor("#333333"),
                    spaceAfter=8
                )

                story = []
                
                # Encabezado del PDF
                story.append(Paragraph("📉 Reporte Ejecutivo de Retención e Inteligencia de Clientes", style_title))
                story.append(Paragraph("Documento automatizado con soporte predictivo de Machine Learning e Inteligencia Artificial (Groq)", style_subtitle))
                story.append(Spacer(1, 10))
                
                # Sección 1: Datos globales recopilados del Dashboard
                story.append(Paragraph("1. Indicadores Base del Negocio (Dashboard)", style_h2))
                txt_kpis = (
                    f"• <b>Clientes en Base de Datos:</b> {kpis['total_customers']:,}<br/>"
                    f"• <b>Tasa de Cancelación Global (Churn Rate):</b> {kpis['churn_rate']:.2f}%<br/>"
                    f"• <b>Facturación Promedio Mensual:</b> ${kpis['avg_monthly']:.2f}<br/>"
                    f"• <b>Ciclo de Vida Medio (Tenure):</b> {kpis['avg_tenure']:.1f} meses"
                )
                story.append(Paragraph(txt_kpis, style_body))
                story.append(Spacer(1, 10))
                
                # Sección 2: El cliente simulado objeto de estudio
                story.append(Paragraph("2. Parámetros del Cliente Evaluado en la Simulación", style_h2))
                data_attr = sim_data['data']
                txt_sim = (
                    f"• <b>Es Adulto Mayor:</b> {'Sí' if data_attr['SeniorCitizen']==1 else 'No'}<br/>"
                    f"• <b>Tipo de Contrato:</b> {data_attr['Contract']}<br/>"
                    f"• <b>Servicio de Internet:</b> {data_attr['InternetService']}<br/>"
                    f"• <b>Soporte Técnico Especializado:</b> {data_attr['TechSupport']}<br/>"
                    f"• <b>Antigüedad del Cliente:</b> {data_attr['tenure']} meses<br/>"
                    f"• <b>Facturación Mensual Estipulada:</b> ${data_attr['MonthlyCharges']:.2f}<br/>"
                    f"• <b>PROBABILIDAD PREDICHA DE FUGA:</b> <b>{sim_data['probability']*100:.2f}%</b><br/>"
                    f"• <b>CLASIFICACIÓN ESTRATÉGICA DEL RIESGO:</b> <b>{sim_data['risk_level']}</b>"
                )
                story.append(Paragraph(txt_sim, style_body))
                story.append(Spacer(1, 15))
                
                # Sección 3: El desglose inteligente de la IA
                story.append(Paragraph("3. Dictamen Consultivo Generado por Groq AI", style_h2))
                
                # Reemplazar saltos de línea planos por saltos legibles en ReportLab HTML paragraphs
                paragraphs_texto_ia = texto_ia.split('\n')
                for p_text in paragraphs_texto_ia:
                    if p_text.strip():
                        # Limpiar sintaxis Markdown que la IA devuelva y que ReportLab no entienda nativamente
                        clean_text = (
                        p_text
                        .replace("**", "")
                        .replace("__", "")
                        .replace("*", "")
                        )
                        story.append(Paragraph(clean_text, style_body))
                
                # Construir el PDF
                doc.build(story)
                buffer.seek(0)
                return buffer

            # Generar los bytes del PDF de ReportLab
            pdf_data = exportar_informe_pdf(texto_informe_groq, kpis_negocio, sim)

            # Botón de descarga automática nativo de Streamlit
            st.download_button(
                label="📥 Descargar Informe Completo en PDF",
                data=pdf_data,
                file_name="Informe_Ejecutivo_Smart_Churn.pdf",
                mime="application/pdf"
            )