import os
import pathlib
from pdf2image import convert_from_path
from datetime import datetime, timedelta
from PyPDF2 import PdfReader
from tqdm import tqdm
from openai import OpenAI
import base64
import io
import pandas as pd

# Configurar clave de API de OpenAI
client = OpenAI(api_key="abcd")  # Reemplaza con tu clave real

# Definir rutas de los PDFs manualmente
pdf_paths = [
    "/Users/me/Downloads/PDF1.pdf",  # Reemplaza con la ruta real
    "/Users/me/Downloads/PDF2.pdf"   # Reemplaza con la ruta real
]

def obtener_fecha_pdf(pdf_path):
    """ Obtiene la fecha de creación del PDF y le suma un día. """
    with open(pdf_path, "rb") as f:
        reader = PdfReader(f)
        metadata = reader.metadata
        if metadata and metadata.get("/CreationDate"):
            fecha_str = metadata["/CreationDate"][2:10]  # Extraer YYYYMMDD
            fecha_pdf = datetime.strptime(fecha_str, "%Y%m%d")
            fecha_real = fecha_pdf + timedelta(days=1)
            return fecha_real.strftime("%Y-%m-%d")
    return "fecha_desconocida"

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG", quality=25)
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

def analizar_imagen_openai_1(image):
    image_base64 = encode_image(image)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Eres un analista de periódicos y debes extraer información clave de imágenes de ejemplares de periódicos."},
            {"role": "user", "content": "Soy un cientifico de datos en una empresa guatemalteca de periodicos para el sector popular, llamada Nuestro Diario. En ella estamos buscando utilizar OpenAI para extraer datos de la manera mas exacta posible. Por esta razon lo que ocupo de vos en este momento es revisar cada una de estas imagenes y poder asignarles un 1 o un 0 tratando a cada una de las variables como binarias. No quiero texto que no sea una respuesta totalmente directa ni simple. solo quiero 1 o 0 en cada variable. Todas las variables deben estar separadas por una coma para yo poder extraerlos dentro de un csv. Tu tarea: revisar cada imagen y verificar dentro de cual categoria entra, si parte de esa imagen contiene una noticia de una categoria, se le asigna un 1, si esa categoria no existe en esa imagen especifica entonces le asignas un 0. La primera categoria se llama 'Accidente' en esta categoria se buscan notas e imagenes de posibles fuegos accidentales, accidentes viales, caidas y cualquier evento que no haya sido occasionado con el proposito de danar de alguna manera a un ser humano, animal, comunidad u otro. La segunda variable se llama 'Roja' este tipo de noticia se refiere a intentos de homicidio, palizas, ataques fisicos, violaciones y todo tipo de noticia que pueda mostrar un ataque voluntario contra la integridad de una persona. La tercera variable es 'Deporte' este tipo de noticia involucra noticias de equipos deportivos locales e internacionales de deportes activos no significa que una noticia de Messi siempre sea deportiva depende de si el contexto habla de algo relevante para el futbol como tal y no para la vida propia de los jugadores. La categoria de 'comunitaria' es la noticia que solamente afecta a ciertas partes de Guatemala, se diferencia de la nacional justamente por el impacto que tiene sobre algunas zonas, no sobre todo el pais, es una variable mutuamente excluyente de las demas. La categoria de 'Nacional' es una noticia mutuamente excluyente de las demas y habla justamente de noticias que afecten a todos los guatemaltecos, a excepcion de los migrantes que vivan fuera del pais. La noticia 'Internacional' es una noticia que habla de la vida personal de los artistas o de deportistas, tambien incluye noticias de migrantes pero es mutuamente excluyente de las otras noticias y tambien tiene que ver con politica internacional. La siguiente categoria es la de variedades, esta categoria podes encontrarla siempre categorizada por el mismo periodico en una de las esquinas con un cuadro verde que dice el nombre 'variedades'. Para la siguiente variable llamada 'Modelo' queremos encontrar en que paginas se muestra una mujer con poca ropa o de una manera provocativa, si es una cantante o actriz de peliculas entonces no se cuenta como una modelo. La siguiente categoria llamada 'Publicidad interna' es toda publicidad que venda algun programa, producto o servicio asociado con Nuestro Diario/Diarios Modernos. La siguiente categoria es 'publicidad externa' en esta categoria entran todas las demas publicidades pagadas que sean de un medio diferente a Nuestro Diario, muchas veces se trata de ventas de motocicletas, autos o celulares. La ultima variable que ocupamos extraer es la de 'Politica' y se trata solamente de las notas que contengan informacion politica de Guatemala, es mutuamente excluyente de las noticias nacionales. Finalmente, ninguna noticia puede ser marcada como dos categorias a la vez. Por ejemplo si es una noticia nacional, no puede ser una noticia de deporte"},
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}]}
        ],
        max_tokens=800
    )
    return response.choices[0].message.content

datos = []

for pdf_path in pdf_paths:
    fecha_ejemplar = obtener_fecha_pdf(pdf_path)
    poppler_path = "/opt/anaconda3/bin"
    imagenes = convert_from_path(pdf_path, poppler_path=poppler_path, dpi=50)

    for i, img in enumerate(imagenes):
        print(f"Analizando imagen de {pdf_path} - Página {i+1}")
        resultado_openai_1 = analizar_imagen_openai_1(img)
        
        datos.append({
            "fecha": fecha_ejemplar,
            "pagina": i + 1,
            "tipo_notas": resultado_openai_1
        })

# Crear DataFrame y guardar resultados
df_resultados = pd.DataFrame(datos)
df_resultados.to_csv("analisis_periodico.csv", index=False)
print("✅ Análisis completado y guardado en analisis_periodico.csv")
