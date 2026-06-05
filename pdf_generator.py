# Importa BytesIO para crear archivos PDF en memoria
# sin necesidad de guardarlos físicamente en disco.
from io import BytesIO

# Importa los componentes principales de ReportLab
# para construir documentos PDF.
from reportlab.platypus import (
    SimpleDocTemplate,  # Documento PDF básico
    Paragraph,          # Párrafos de texto
    Spacer              # Espacios en blanco entre elementos
)

# Importa estilos predefinidos para dar formato al PDF.
from reportlab.lib.styles import (
    getSampleStyleSheet
)


# ==========================================================
# Función Generadora de PDF
# ==========================================================

def build_pdf(text):
    """
    Genera un archivo PDF a partir de un texto recibido.

    Parámetros:
        text (str): contenido que se incluirá en el PDF.

    Retorna:
        bytes: archivo PDF listo para descarga.
    """

    # Crea un buffer temporal en memoria donde se construirá el PDF.
    buffer = BytesIO()

    # Crea el documento PDF utilizando el buffer como destino.
    doc = SimpleDocTemplate(buffer)

    # Obtiene una colección de estilos predeterminados de ReportLab.
    styles = getSampleStyleSheet()

    # Lista que almacenará todos los elementos del PDF.
    content = []

    # ======================================================
    # Título del Documento
    # ======================================================

    # Agrega un título principal al PDF.
    content.append(
        Paragraph(
            "Informe Ejecutivo Inteligente",
            styles["Title"]  # Estilo de título predefinido
        )
    )

    # Inserta un espacio vertical de 12 puntos
    # después del título.
    content.append(
        Spacer(1, 12)
    )

    # ======================================================
    # Contenido del Informe
    # ======================================================

    # Divide el texto recibido en líneas utilizando
    # el salto de línea como separador.
    for line in text.split("\n"):

        # Verifica que la línea no esté vacía.
        if line.strip():

            # Agrega la línea como un párrafo al PDF.
            content.append(
                Paragraph(
                    line,
                    styles["BodyText"]  # Estilo de texto normal
                )
            )

    # ======================================================
    # Construcción del PDF
    # ======================================================

    # Genera físicamente el documento PDF utilizando
    # todos los elementos almacenados en la lista content.
    doc.build(content)

    # Obtiene el contenido binario completo del PDF.
    pdf = buffer.getvalue()

    # Libera la memoria utilizada por el buffer.
    buffer.close()

    # Devuelve el archivo PDF en formato bytes.
    return pdf