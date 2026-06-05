# Importa la clase Groq necesaria para conectarse
# a la API de inteligencia artificial de Groq.
from groq import Groq

# Importa load_dotenv para cargar variables
# de entorno almacenadas en un archivo .env.
from dotenv import load_dotenv

# Importa el módulo os para acceder a variables
# del sistema operativo.
import os


# ==========================================================
# Carga de Variables de Entorno
# ==========================================================

# Lee automáticamente el archivo .env ubicado
# en la raíz del proyecto.
load_dotenv()


# ==========================================================
# Configuración del Cliente GROQ
# ==========================================================

# Crea una instancia del cliente Groq utilizando
# la clave API almacenada en la variable de entorno.
client = Groq(

    api_key=os.getenv(
        "GROQ_API_KEY"
    )
)


# ==========================================================
# Generación de Informes Ejecutivos
# ==========================================================

def generate_executive_report(prompt):
    """
    Envía un prompt a Groq y devuelve
    un informe ejecutivo generado por IA.

    Parámetros:
        prompt (str): instrucción o contexto que
                      será enviado al modelo.

    Retorna:
        str: respuesta generada por la IA.
    """

    # ======================================================
    # Solicitud al Modelo
    # ======================================================

    response = client.chat.completions.create(

        # Modelo LLM utilizado
        model="llama-3.3-70b-versatile",

        # Conversación enviada al modelo
        messages=[

            {
                # Rol del mensaje
                "role": "user",

                # Contenido del prompt
                "content": prompt
            }

        ],

        # Controla el nivel de creatividad.
        # Valores bajos generan respuestas más consistentes.
        temperature=0.3
    )

    # ======================================================
    # Respuesta
    # ======================================================

    # Extrae únicamente el texto generado
    # por el modelo.
    return (
        response
        .choices[0]
        .message
        .content
    )