import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer, AutoModel
from transformers import AutoModelForSequenceClassification
from transformers import BertTokenizer, BertForSequenceClassification, BertModel
import torch

df = pd.read_csv("Datos_Portadas.csv")
data = pd.DataFrame(df)

vectorizer = CountVectorizer(max_features=5000, stop_words = spanish_stopwords)

# Fill NaN values with an empty string
data["titulo"] = data["titulo"].fillna("").astype(str)
data["subtitular"] = data["subtitular"].fillna("").astype(str)

data["text"] = data["titulo"] + " " + data["subtitular"]

# Define range of parameters to test
n_components_range = [5, 10, 15, 20, 25, 30, 35, 40]  # Example LDA n_components values
num_classes_range = [5, 10, 15, 20]  # Example LCA num_classes values

# Prepare an empty list to store the latent class distributions
distributions = []

for n_components in n_components_range:
    for num_classes in num_classes_range:
        # Apply LDA with Spanish stop words
        vectorizer = CountVectorizer(max_features=5000, stop_words=spanish_stopwords)
        X = vectorizer.fit_transform(data["text"])

        lda = LatentDirichletAllocation(n_components=n_components, random_state=42)
        X_topics = lda.fit_transform(X)
        data["topic"] = X_topics.argmax(axis=1)

        # Apply OneHotEncoder
        encoder = OneHotEncoder(sparse_output=False)
        X_lca = encoder.fit_transform(data[["topic"]])

        # Apply LCA (GaussianMixture)
        lca_model = GaussianMixture(n_components=num_classes, covariance_type="full", random_state=42)
        data["latent_class"] = lca_model.fit_predict(X_lca)

        # Calculate class distribution
        class_distribution = data["latent_class"].value_counts().values
        distributions.append([n_components, num_classes, class_distribution])

# Convert results into a DataFrame for visualization
distributions_df = pd.DataFrame(distributions, columns=["n_components", "num_classes", "class_distribution"])

# Make sure all classes are represented in each distribution
distributions_df["class_count"] = distributions_df["class_distribution"].apply(lambda x: np.sum(x))

scaler = StandardScaler()
X_lca_scaled = scaler.fit_transform(X_lca)

# Define number of latent classes
num_classes = 19

# Fit LCA model using Gaussian Mixture Model (GMM)
lca_model = GaussianMixture(n_components=num_classes, covariance_type="spherical", init_params="kmeans", random_state=20)
data["latent_class"] = lca_model.fit_predict(X_lca)

# Check class distribution
print(data["latent_class"].value_counts())

# Get feature names (words)
feature_names = vectorizer.get_feature_names_out()

# Function to extract top words per topic
def get_top_words(lda_model, feature_names, num_words=50):
    topic_keywords = {}
    for topic_idx, topic in enumerate(lda_model.components_):
        top_keywords = [feature_names[i] for i in topic.argsort()[-num_words:]]  
        topic_keywords[topic_idx] = top_keywords
    return topic_keywords

# Get words for each topic
topic_keywords = get_top_words(lda, feature_names)

# Display top words per topic
for topic, words in topic_keywords.items():
    print(f"Topic {topic}: {', '.join(words)}")

new_title = "taxis pirata estan en la mira"
new_subtitle = "autoridades buscan diezmar inseguridad y criminalidad"

# Merge text
new_text = new_title + " " + new_subtitle

# Transform into vector
X_new = vectorizer.transform([new_text])
X_new_topic = lda.transform(X_new).argmax(axis=1)  # Get topic

# Convert topic into categorical format
X_new_lca = encoder.transform([[X_new_topic[0]]])

# Predict latent class
predicted_class = lca_model.predict(X_new_lca)
print("Predicted Latent Class:", predicted_class[0])

# Load BETO model and tokenizer
beto_model_name = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = AutoTokenizer.from_pretrained(beto_model_name)
model = AutoModel.from_pretrained(beto_model_name)

# Function to extract BETO embeddings
def get_beto_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Mean pooling
    return embedding

# Ensure embeddings are stored in data
data["beto_embedding"] = data.apply(lambda row: get_beto_embedding(row["titulo"] + " " + row["subtitular"]), axis=1)

# Convert to array
X_beto = np.vstack(data["beto_embedding"].values)

# Scale embeddings
scaler = StandardScaler()
X_beto_scaled = scaler.fit_transform(X_beto)

# Fit Gaussian Mixture Model
num_classes = 25
lca_model = GaussianMixture(
    n_components=num_classes,
    covariance_type="tied", 
    init_params="kmeans",
    n_init=10,
    random_state=20
)
data["latent_class"] = lca_model.fit_predict(X_beto_scaled)

# Print unique classes to verify all 19 are assigned
print("Unique classes:", sorted(data["latent_class"].unique()))
print(data["latent_class"].value_counts())

# Save results
data[['titulo', 'subtitular', 'predicted_category', 'latent_class']].to_csv("predicted_categories_with_classes.csv", index=False)
print("CSV saved successfully!")

category_mapping = {0: 'Obras Infraestructura', 1: 'Crimenes', 2: 'Mejoras Comunitarias', 3: 'Accidentes y Rojas', 4: 'Seguridad y Comercio', 5: 'Crimen Hogar', 6: 'Peticiones Comunitarias', 7: 'Obras Infraestructuras', 8: 'Comunitarios Vial', 9: 'Salud y Agua', 10: 'Mejoras Comunitarias', 11: 'Accidentes Familiares', 12: 'Delincuencia', 13: 'Emergencias Comunitarias', 14: 'Precaucion en Comunidades', 15: 'Seguridad y Comercio', 16: 'Eventos Nacionales', 17: 'Clases', 18: 'Delincuencia'}
data['predicted_category_name'] = data['predicted_category'].map(category_mapping)
print(data[['predicted_category_name']])

# Count occurrences of each (latent_class, predicted_category) pair
class_topic_counts = data.groupby(["latent_class", "predicted_category"]).size().reset_index(name="count")

# Normalize counts to get topic proportions within each latent class
class_topic_counts["proportion"] = class_topic_counts.groupby("latent_class")["count"].transform(lambda x: x / x.sum())

# Sort to see the most relevant topics per latent class
class_topic_counts = class_topic_counts.sort_values(by=["latent_class", "proportion"], ascending=[True, False])

print(class_topic_counts)

# Function to assign categories based on top words
def assign_category_to_latent_class(latent_class, topic_keywords):
    # Get the top words for the corresponding latent class
    top_words = topic_keywords.get(latent_class, [])
        # Based on the top words, you can manually map categories (or you can use a predefined set of rules)
    category_mapping = {
        "Seguridad y Comercio": ["logra", "andres", "persona", "sistema", "reportan", "centro", "varios", "afecta", "calle", "millones", "piden", "seguridad", "comunidad", "aumentan", "desnutricion", "personas", "pagan", "luis", "ayampuc", "escuelas", "colonia", "educativos", "antonio", "tuberia", "agua", "francisco", "ninos", "dengue", "menos", "flores", "obra", "salud", "40", "cada", "organizan", "jose", "semanas", "vehiculos", "casos", "esta", "trabajos", "aldea", "municipios", "centros", "autoridades", "afectados", "vecinos", "sacatepequez", "marcos", "san"],
        "Delincuencia y Actividad Criminal": ["colision", "perdidas", "asesino", "alcaldes", "poblacion", "vecinos", "lluvias", "viaje", "ni", "pueden", "acciones", "municipio", "camino", "piden", "docentes", "frente", "menos", "fallece", "disparo", "partido", "familia", "hombres", "disparos", "bebe", "herido", "dispara", "peten", "ediles", "tragedia", "falta", "ingreso", "cierran", "flores", "afectado", "personas", "trabajo", "ruta", "autoridades", "mueren", "ataque", "mientras", "hecho", "ladron", "presunto", "paso", "hombre", "cementerio", "victima", "esta", "matan"],
        "Salud y Bienestar Comunitario": ["chimaltenango", "pnc", "comienzan", "parte", "esperan", "casa", "cuaresma", "pobladores", "trabajadores", "todos", "acompanan", "otros", "viven", "libra", "manana", "inicio", "llenan", "caserio", "comerciante", "tiros", "rodeo", "faltan", "medida", "arrastra", "san", "hoy", "asesinados", "clases", "familias", "salud", "socorristas", "mayor", "cadaver", "casas", "pandilleros", "policia", "atender", "viviendas", "centros", "visitan", "rio", "sufre", "muerte", "hallan", "aumento", "llegan", "alumnos", "preparan", "ciclo", "escolar"],
        "Desarrollo Comunitario y Mejoras": ["nuevo", "llamas", "cifra", "presuntos", "pais", "80", "pnc", "fallecidos", "cortocircuito", "vecinos", "victima", "departamental", "celebran", "muere", "viaje", "danos", "fuego", "tragedia", "lugar", "unas", "marquense", "siniestro", "pierden", "ataque", "cada", "ciudad", "millones", "san", "calles", "casa", "guatemala", "coyotes", "reportan", "victimas", "muerte", "barranco", "mexico", "mujer", "edad", "familia", "violencia", "denuncias", "menores", "perdidas", "migrantes", "estados", "unidos", "personas", "vida", "incendio"],
        "Infraestructura Pública y Obras Viales": ["rn", "pobladores", "puente", "tierra", "cerca", "alerta", "agricultores", "habilitan", "vida", "capturados", "preven", "nuevas", "hombres", "conecta", "beneficiara", "hectareas", "vehicular", "parque", "esta", "feria", "sobrevive", "dentro", "prados", "costo", "afectan", "autoridades", "derrumbes", "meses", "mes", "cultivos", "villa", "pierden", "balean", "sido", "maiz", "disparan", "hermosa", "zona", "construccion", "obra", "desnivel", "ruta", "lluvias", "proyecto", "municipios", "personas", "ataque", "armado", "millones", "paso"],
        "Accidentes y Tragedias": ["bomberos", "hija", "embestido", "vecinos", "fuga", "grave", "microbus", "huye", "chocar", "cadaver", "vehiculos", "estaba", "casa", "quedan", "metros", "mano", "transito", "arrolla", "choque", "heridas", "volcar", "ruta", "personas", "ser", "hombre", "arbol", "herido", "hondonada", "bus", "piloto", "horas", "impacto", "vuelca", "zona", "muerto", "camion", "trailer", "causa", "mujer", "vehiculo", "cae", "picop", "control", "percance", "conductor", "muere", "chofer", "accidente", "heridos", "restos"],
        "Accidentes Familiares": ["capital", "reportan", "acribillan", "matan", "muerte", "patrono", "recorre", "hechos", "maria", "centro", "asesinatos", "cortejo", "muertes", "imagen", "accidentes", "misa", "registran", "afectados", "autoridades", "fe", "cruz", "drogas", "guatemala", "departamento", "quien", "menos", "historico", "sicarios", "feligreses", "incremento", "muertos", "recorrido", "participan", "cae", "menores", "alerta", "departamentos", "verapaz", "zona", "fieles", "captura", "hombre", "alta", "policia", "procesion", "semana", "dengue", "jesus", "santa", "casos"],
        "Proyectos de Construcción e Infraestructura": ["herido", "mata", "embiste", "ser", "esposa", "caer", "menor", "dirigia", "salen", "salio", "hijo", "nueva", "vivienda", "milagro", "auto", "padre", "salvan", "hijos", "quedan", "muerte", "frenos", "ultimo", "villa", "hija", "vida", "choque", "velocidad", "murio", "familiares", "amigos", "frente", "quedo", "carro", "morir", "iban", "impacta", "casa", "joven", "picop", "mueren", "rio", "madre", "hombre", "accidente", "choca", "familia", "trailer", "vehiculo", "camion", "muere"],
        "Precauciones en las Comunidades": ["derrumbes", "buscan", "comunica", "llenan", "dano", "frontera", "camion", "atlantico", "quebrada", "planta", "hora", "deben", "felipe", "evitar", "villa", "centro", "camino", "nueva", "transportistas", "afecta", "meses", "cierre", "agua", "comunidad", "punto", "santiago", "labores", "minutos", "tratamiento", "transito", "esta", "construccion", "apostol", "pacifico", "comuna", "puente", "ciudad", "honor", "riesgo", "socavamiento", "hundimiento", "aldea", "temen", "colapso", "san", "carril", "carretera", "vecinos", "ruta", "paso"],
        "Crímenes y Actos Delictivos": ["escasez", "motagua", "quedar", "esperan", "carretera", "proyecto", "distribucion", "tomas", "barrio", "afecta", "invierno", "paso", "santo", "calle", "problema", "obras", "incomunicados", "casas", "podria", "aguas", "ruta", "autoridades", "camino", "afectadas", "afectan", "barrios", "puerto", "drenajes", "lodo", "falta", "estado", "llevan", "pozo", "servicio", "meses", "alda", "mal", "inundaciones", "san", "exigen", "piden", "santa", "puente", "lluvias", "comunidades", "pobladores", "rio", "familias", "vecinos", "agua"],
        "Eventos Nacionales y Seguridad Pública": ["terror", "autoridades", "muertos", "masacre", "heridas", "juarez", "hallan", "area", "lluvias", "ser", "pacientes", "ataca", "crisis", "veces", "cuerpos", "casos", "volcan", "casas", "mientras", "mujeres", "mes", "hombre", "familias", "casa", "accidente", "tambien", "habria", "balazos", "joven", "horas", "millones", "siguen", "vida", "espera", "carretera", "sido", "afectadas", "muerte", "buscan", "familiares", "ciudad", "migrantes", "mexico", "incendio", "hospital", "blanca", "menos", "arma", "personas", "fuego"],
        "Peticiones y Solicitudes Comunitarias": ["ruta", "barrio", "otros", "personas", "buses", "coatepeque", "sigue", "sur", "santa", "mejoran", "deben", "muro", "caminos", "transito", "policia", "lluvia", "comuna", "zonas", "construccion", "vehicular", "reparan", "accidentes", "estructura", "metros", "ser", "entrega", "caen", "exigen", "temen", "fin", "municipios", "horas", "colonia", "pasaje", "lluvias", "autoridades", "carga", "sector", "principal", "riesgo", "debido", "avenida", "obra", "meses", "zona", "puente", "paso", "transporte", "calle", "vecinos"],
        "Migrantes e Instituciones": ["calle", "mantienen", "final", "reune", "toman", "reporta", "policial", "piden", "75", "ninos", "esperan", "muere", "mp", "alegria", "visita", "500", "madre", "muertes", "general", "an", "tamales", "sectores", "policia", "hechos", "pasado", "muertos", "hijo", "huehuetenango", "violentas", "hogar", "mama", "bebe", "acompana", "ultimos", "fase", "va", "pais", "enero", "personas", "centro", "comerciantes", "presencia", "hoy", "celebran", "primera", "ciudad", "mayor", "vuelta", "mujer", "segunda"],
        "Comunitarios Vial": ["fuerzas", "interamericana", "kilometro", "estado", "temen", "actividades", "mueren", "heridos", "vecinos", "hectareas", "deslizamientos", "meses", "socorristas", "comunitarios", "valle", "usuarios", "bono", "dejan", "policia", "fiestas", "terminal", "delictivos", "tierra", "periferico", "siniestros", "muerto", "fuego", "sector", "victima", "paso", "rio", "hechos", "evita", "pago", "trabajar", "ruta", "prevenir", "departamento", "guatemala", "policias", "forestales", "vigilan", "retiran", "pnc", "piden", "asaltos", "colision", "incendios", "agentes", "seguridad"],
        "Emergencias Nacionales y Seguridad": ["balean", "temperaturas", "familias", "varias", "cada", "familiares", "cabeza", "ser", "ellas", "sufren", "vacaciones", "ofrecer", "joven", "mujer", "celebran", "bloqueos", "cero", "albanil", "visita", "80", "destruye", "veraneantes", "motoristas", "horas", "lleva", "carreteras", "comunidad", "mueren", "600", "santa", "productos", "alza", "menos", "nueva", "corpus", "desconocidos", "ciudad", "nuevo", "asesinado", "christi", "puente", "meses", "esperan", "salud", "cerca", "semana", "precios", "bajo", "aprovechan", "centro"],
        "Desarrollo y Planificación de Infraestructuras": ["atencion", "nacional", "joven", "cierran", "danos", "acceso", "construccion", "recibiran", "transito", "pasar", "pueblo", "punto", "60", "meses", "desaparecido", "regresaba", "santa", "chiquimula", "camino", "autoridades", "jornada", "finalizar", "cristo", "trailer", "estaba", "instalaciones", "muerte", "personas", "casa", "estuvo", "hacia", "fallece", "atropellado", "fin", "esquipulas", "ruta", "semana", "vecinos", "exigen", "carretera", "reparacion", "negro", "pacientes", "alda", "puente", "paso", "arrollado", "hombre", "muere", "hospital"],
        "Delincuencia y Tendencias Criminales": ["clausura", "inician", "bajo", "fundacion", "primer", "cientos", "fallecido", "pan", "guadalupe", "guatemala", "grupo", "recibir", "miles", "fiestas", "hallado", "asuncion", "mundo", "vez", "reciben", "municipio", "semana", "personas", "rosario", "sube", "cadaver", "patrona", "catolicos", "precio", "cantan", "celebra", "mes", "fe", "involucrados", "salud", "dona", "cierra", "carretera", "directo", "aumento", "ruta", "fiesta", "fiestas", "victimas", "guatemala", "piden", "grupos"],
        "Crímenes en el Hogar y Robos": ["accidente", "camion", "unico", "grupo", "victimas", "bloqueo", "ser", "hombres", "muertos", "zona", "municipio", "carreteras", "heridos", "guatemala", "autoridades", "camino", "grande", "defensa", "vuelta", "exigen", "llama", "cooperativa", "tiene", "sufren", "herida", "policiais", "derrumbe", "sismo", "muere", "tragedia", "paso", "familia", "trailer", "veces", "vivo", "accidentes", "rescate", "tiempo", "informe", "delito", "policia", "transito", "traslado", "hombres", "construccion", "ciclo", "hospital", "responsables", "cuidad", "niños", "graves", "nuevas", "carretera"],
        "Clases e Iniciativas Educativas": ["tormenta", "hospital", "afecta", "hombres", "servicio", "estadio", "puente", "ofrecen", "municipios", "personas", "desfiles", "comienza", "itza", "material", "cementerio", "nacional", "san", "torneo", "primera", "potable", "poblacion", "paso", "alerta", "cabecera", "peten", "carretera", "pais", "area", "visitas", "departamento", "covid", "millones", "zonas", "esta", "suspenden", "autoridades", "esperan", "costara", "atencion", "ruta", "rios", "solo", "puesto", "comunidades", "construccion", "agua", "lago", "meses", "centro", "salud"],
        "Emergencias y Respuesta a Crisis": ["chiquimula", "temen", "estaban", "cerca", "detienen", "pasajeros", "chofer", "desconocidos", "agricultor", "aldea", "transito", "llegar", "victimas", "caminaba", "robos", "asaltantes", "bala", "menor", "vecinos", "victima", "salir", "hombres", "joven", "banda", "robo", "ninos", "dejar", "asesinan", "barranco", "eeuu", "mientras", "ayuda", "escapar", "piden", "garcia", "familia", "delincuentes", "vivienda", "zona", "atacan", "muere", "meses", "disparan", "cae", "balazos", "hombre", "vida", "mujer", "atacado", "casa"],
        "Conciencia Pública y Comunicación": ["orfandad", "ninas", "conmemoran", "catolicos", "muerte", "vuelven", "remodelacion", "rafael", "plantel", "estudian", "nuevo", "cientos", "ciudad", "jubilo", "alegria", "casa", "dios", "alumnas", "estudiar", "region", "devocion", "construir", "fondos", "terminar", "nino", "salones", "regresan", "tradicion", "quieren", "hijos", "celebran", "iglesia", "calles", "escuelas", "aldea", "falta", "san", "esta", "exigen", "techo", "reciben", "ninos", "aulas", "familia", "maestros", "alumnos", "padres", "estudiantes", "clases", "escuela"],
        "Seguridad y Regulación del Comercio": ["agresion", "colonia", "balas", "disputa", "local", "hospital", "ver", "asaltos", "sicarios", "pareja", "moto", "camino", "edad", "ayer", "asesinado", "tambien", "disparos", "mientras", "tenia", "asesinan", "acusan", "balazos", "mareros", "sexual", "frente", "robo", "centro", "asesinato", "sector", "pnc", "violacion", "ser", "zona", "matar", "hombres", "barrio", "ladrones", "menor", "supuestos", "capturado", "delincuentes", "extorsion", "disparan", "asalto", "mujer", "victimas", "hombre", "matan", "capturan", "mujeres"],
        "Mejoras y Bienestar Comunitario": ["casco", "canales", "porque", "casi", "hacia", "rutas", "vehiculos", "turismo", "paso", "cuenta", "continuan", "villa", "hay", "pilotos", "invierno", "calle", "afecta", "reparar", "denuncian", "semana", "flores", "mientras", "interamericana", "baches", "personas", "solo", "urbano", "abandono", "kilometros", "comienza", "trabajos", "reparacion", "comercio", "transportistas", "comerciantes", "viven", "accidentes", "exigen", "piden", "san", "festival", "via", "ciudad", "tramo", "autoridades", "visitantes", "turistas", "esperan", "ruta", "vecinos"],
        "Fiestas Nacionales": ["temen", "comuna", "peregrinacion", "nueva", "llego", "nacional", "mes", "municipio", "podrian", "residentes", "encuentran", "pide", "carcel", "mal", "usar", "independencia", "muerte", "espera", "lluvias", "unidos", "personal", "estado", "chimaltenango", "aseguran", "aldea", "recibe", "ingreso", "40", "camino", "solo", "lleva", "hospital", "esperaron", "alternas", "vias", "piden", "edificio", "nuevo", "horas", "muertos", "sido", "meses", "proyecto", "pavimentacion", "abandono", "tiene", "puente", "trabajos", "obra", "vecinos"],
        "Salud Pública y Gestión del Agua": ["calles", "negocio", "mes", "arranca", "hombre", "tierra", "celebran", "olores", "cena", "jorge", "desborde", "rio", "cae", "afecta", "otras", "obra", "comunidades", "esta", "riesgo", "aguas", "metros", "bienvenida", "comuna", "guatemala", "arrancan", "maestro", "afectadas", "150", "nuevo", "policia", "zona", "hipico", "beneficiadas", "antigua", "esperan", "plaza", "jose", "fin", "ninos", "fiestas", "mercado", "ciudad", "drenaje", "actividades", "calle", "vecinos", "familias", "san", "feria", "desfile"]
    }

    # Find a matching category based on top words
    for category, keywords in category_mapping.items():
        if any(word in top_words for word in keywords):
            return category
    
    return "Otros"  # If no match, assign a default category

# Apply category assignment based on the top words of each latent class
data["predicted_category_name_from_words"] = data["latent_class"].apply(
    lambda x: assign_category_to_latent_class(x, topic_keywords)
)

# Print the updated dataframe with predicted category names
print(data[['titulo', 'subtitular', 'latent_class', 'predicted_category_name_from_words']])

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

# Function to extract BETO embeddings with tokenizer and model as arguments
def get_beto_embedding(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Mean pooling
    return embedding


# Function to predict latent class for a new title and subtitle using BERT embeddings
def predict_latent_class_with_beto(new_title, new_subtitle, vectorizer, lda_model, encoder, lca_model, tokenizer, beto_model, scaler):
    # Merge title and subtitle into one text string
    new_text = new_title + " " + new_subtitle

    # Get the BERT embedding for the new text
    new_embedding = get_beto_embedding(new_text, tokenizer, beto_model)

    # Standardize the new embedding using the same scaler used during training
    new_embedding_scaled = scaler.transform([new_embedding])

    # Predict the latent class using the Gaussian Mixture Model (GMM)
    predicted_class = lca_model.predict(new_embedding_scaled)

    # Return the predicted latent class
    return predicted_class[0]


# Function to find the nearest valid latent class
def find_nearest_class(predicted_class, valid_classes):
    # Calculate the Euclidean distance from the predicted class to each valid class
    distances = euclidean_distances(np.array([[predicted_class]]), np.array([[cls] for cls in valid_classes]))
    
    # Find the index of the closest valid class
    nearest_class_index = np.argmin(distances)
    
    # Return the nearest valid class
    return valid_classes[nearest_class_index]


# Example new title and subtitle
new_title = "maestra se opone a asalto y le disparan"
new_subtitle = "delincuentes iban en moto se le cruzan en la ruta y abren fuego profesora"

# Predict the latent class for the new title and subtitle
predicted_class = predict_latent_class_with_beto(new_title, new_subtitle, vectorizer, lda, encoder, lca_model, tokenizer, model, scaler)

# Define the allowed classes (indices)
category_mapping = {
    15: 'Obras Infraestructura', 9: 'Crimenes', 22: 'Mejoras Comunitarias', 5: 'Accidentes y Rojas', 
    21: 'Seguridad y Comercio', 17: 'Crimen Hogar', 11: 'Peticiones Comunitarias', 7: 'Obras Infraestructuras', 
    4: 'Comunitarios Vial', 24: 'Salud y Agua', 3: 'Mejoras Comunitarias', 6: 'Accidentes Familiares', 
    16: 'Delincuencia', 19: 'Emergencias Comunitarias', 8: 'Precaucion en Comunidades', 
    0: 'Seguridad y Comercio', 10: 'Eventos Nacionales', 18: 'Clases', 1: 'Delincuencia'
}

# Get the valid class indices
valid_classes = list(category_mapping.keys())

# Check if the predicted class is in the valid classes
if predicted_class in valid_classes:
    # Get the category name based on the predicted class
    predicted_category_name = category_mapping.get(predicted_class, "Unknown")
    print("Predicted Category:", predicted_category_name)
else:
    # Find the nearest valid latent class
    nearest_class = find_nearest_class(predicted_class, valid_classes)
    
    # Get the category name for the nearest valid class
    predicted_category_name = category_mapping.get(nearest_class, "Unknown")
    print("Predicted Category (nearest class):", predicted_category_name)


import torch
from transformers import AutoTokenizer, AutoModel
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import BayesianGaussianMixture
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# Load BETO model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")
model = AutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-cased")

# Load the dataset
df = pd.read_csv("your_dataset.csv")  # Ensure this contains a column with text data
text_column = "headline"  # Change this to your actual text column

# Tokenization and embedding
embeddings = []
for text in df[text_column].astype(str).tolist():
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    
    cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # Use CLS token
    embeddings.append(cls_embedding)

X_beto = np.array(embeddings)

# Standardize embeddings
scaler = StandardScaler()
X_beto_scaled = scaler.fit_transform(X_beto)

# Visualize with t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_beto_scaled)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.5)
plt.title("t-SNE Visualization of BETO Embeddings")
plt.show()

# Bayesian Gaussian Mixture Model (BGMM)
bgmm = BayesianGaussianMixture(n_components=30, covariance_type="tied", random_state=42)
df["cluster"] = bgmm.fit_predict(X_beto_scaled)

# Save results
df.to_csv("clustered_results.csv", index=False)


//////////
/////////

/////////

//////////


# Load BETO model and tokenizer
beto_model_name = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = AutoTokenizer.from_pretrained(beto_model_name)
model = AutoModel.from_pretrained(beto_model_name)

# Function to extract BETO embeddings
def get_beto_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()  # Mean pooling
    return embedding

# Ensure embeddings are stored in data
data["beto_embedding"] = data.apply(lambda row: get_beto_embedding(row["titulo"] + " " + row["subtitular"]), axis=1)

# Convert to array
X_beto = np.vstack(data["beto_embedding"].values)

# Scale embeddings
scaler = StandardScaler()
X_beto_scaled = scaler.fit_transform(X_beto)

# Fit Gaussian Mixture Model
num_classes = 19
lca_model = GaussianMixture(
    n_components=num_classes,
    covariance_type="tied", 
    init_params="kmeans",
    n_init=10,
    random_state=20
)
data["latent_class"] = lca_model.fit_predict(X_beto_scaled)

# Print unique classes to verify all 19 are assigned
print("Unique classes:", sorted(data["latent_class"].unique()))
print(data["latent_class"].value_counts())

# Function to assign categories based on top words
def assign_category_to_latent_class(latent_class, topic_keywords):
    # Get the top words for the corresponding latent class
    top_words = topic_keywords.get(latent_class, [])
        # Based on the top words, you can manually map categories (or you can use a predefined set of rules)
    category_mapping = {
        "Seguridad y Comercio": ["logra", "andres", "persona", "sistema", "reportan", "centro", "varios", "afecta", "calle", "millones", "piden", "seguridad", "comunidad", "aumentan", "desnutricion", "personas", "pagan", "luis", "ayampuc", "escuelas", "colonia", "educativos", "antonio", "tuberia", "agua", "francisco", "ninos", "dengue", "menos", "flores", "obra", "salud", "40", "cada", "organizan", "jose", "semanas", "vehiculos", "casos", "esta", "trabajos", "aldea", "municipios", "centros", "autoridades", "afectados", "vecinos", "sacatepequez", "marcos", "san"],
        "Delincuencia y Actividad Criminal": ["colision", "perdidas", "asesino", "alcaldes", "poblacion", "vecinos", "lluvias", "viaje", "ni", "pueden", "acciones", "municipio", "camino", "piden", "docentes", "frente", "menos", "fallece", "disparo", "partido", "familia", "hombres", "disparos", "bebe", "herido", "dispara", "peten", "ediles", "tragedia", "falta", "ingreso", "cierran", "flores", "afectado", "personas", "trabajo", "ruta", "autoridades", "mueren", "ataque", "mientras", "hecho", "ladron", "presunto", "paso", "hombre", "cementerio", "victima", "esta", "matan"],
        "Salud y Bienestar Comunitario": ["chimaltenango", "pnc", "comienzan", "parte", "esperan", "casa", "cuaresma", "pobladores", "trabajadores", "todos", "acompanan", "otros", "viven", "libra", "manana", "inicio", "llenan", "caserio", "comerciante", "tiros", "rodeo", "faltan", "medida", "arrastra", "san", "hoy", "asesinados", "clases", "familias", "salud", "socorristas", "mayor", "cadaver", "casas", "pandilleros", "policia", "atender", "viviendas", "centros", "visitan", "rio", "sufre", "muerte", "hallan", "aumento", "llegan", "alumnos", "preparan", "ciclo", "escolar"],
        "Desarrollo Comunitario y Mejoras": ["nuevo", "llamas", "cifra", "presuntos", "pais", "80", "pnc", "fallecidos", "cortocircuito", "vecinos", "victima", "departamental", "celebran", "muere", "viaje", "danos", "fuego", "tragedia", "lugar", "unas", "marquense", "siniestro", "pierden", "ataque", "cada", "ciudad", "millones", "san", "calles", "casa", "guatemala", "coyotes", "reportan", "victimas", "muerte", "barranco", "mexico", "mujer", "edad", "familia", "violencia", "denuncias", "menores", "perdidas", "migrantes", "estados", "unidos", "personas", "vida", "incendio"],
        "Infraestructura Pública y Obras Viales": ["rn", "pobladores", "puente", "tierra", "cerca", "alerta", "agricultores", "habilitan", "vida", "capturados", "preven", "nuevas", "hombres", "conecta", "beneficiara", "hectareas", "vehicular", "parque", "esta", "feria", "sobrevive", "dentro", "prados", "costo", "afectan", "autoridades", "derrumbes", "meses", "mes", "cultivos", "villa", "pierden", "balean", "sido", "maiz", "disparan", "hermosa", "zona", "construccion", "obra", "desnivel", "ruta", "lluvias", "proyecto", "municipios", "personas", "ataque", "armado", "millones", "paso"],
        "Accidentes y Tragedias": ["bomberos", "hija", "embestido", "vecinos", "fuga", "grave", "microbus", "huye", "chocar", "cadaver", "vehiculos", "estaba", "casa", "quedan", "metros", "mano", "transito", "arrolla", "choque", "heridas", "volcar", "ruta", "personas", "ser", "hombre", "arbol", "herido", "hondonada", "bus", "piloto", "horas", "impacto", "vuelca", "zona", "muerto", "camion", "trailer", "causa", "mujer", "vehiculo", "cae", "picop", "control", "percance", "conductor", "muere", "chofer", "accidente", "heridos", "restos"],
        "Accidentes Familiares": ["capital", "reportan", "acribillan", "matan", "muerte", "patrono", "recorre", "hechos", "maria", "centro", "asesinatos", "cortejo", "muertes", "imagen", "accidentes", "misa", "registran", "afectados", "autoridades", "fe", "cruz", "drogas", "guatemala", "departamento", "quien", "menos", "historico", "sicarios", "feligreses", "incremento", "muertos", "recorrido", "participan", "cae", "menores", "alerta", "departamentos", "verapaz", "zona", "fieles", "captura", "hombre", "alta", "policia", "procesion", "semana", "dengue", "jesus", "santa", "casos"],
        "Proyectos de Construcción e Infraestructura": ["herido", "mata", "embiste", "ser", "esposa", "caer", "menor", "dirigia", "salen", "salio", "hijo", "nueva", "vivienda", "milagro", "auto", "padre", "salvan", "hijos", "quedan", "muerte", "frenos", "ultimo", "villa", "hija", "vida", "choque", "velocidad", "murio", "familiares", "amigos", "frente", "quedo", "carro", "morir", "iban", "impacta", "casa", "joven", "picop", "mueren", "rio", "madre", "hombre", "accidente", "choca", "familia", "trailer", "vehiculo", "camion", "muere"],
        "Precauciones en las Comunidades": ["derrumbes", "buscan", "comunica", "llenan", "dano", "frontera", "camion", "atlantico", "quebrada", "planta", "hora", "deben", "felipe", "evitar", "villa", "centro", "camino", "nueva", "transportistas", "afecta", "meses", "cierre", "agua", "comunidad", "punto", "santiago", "labores", "minutos", "tratamiento", "transito", "esta", "construccion", "apostol", "pacifico", "comuna", "puente", "ciudad", "honor", "riesgo", "socavamiento", "hundimiento", "aldea", "temen", "colapso", "san", "carril", "carretera", "vecinos", "ruta", "paso"],
        "Crímenes y Actos Delictivos": ["escasez", "motagua", "quedar", "esperan", "carretera", "proyecto", "distribucion", "tomas", "barrio", "afecta", "invierno", "paso", "santo", "calle", "problema", "obras", "incomunicados", "casas", "podria", "aguas", "ruta", "autoridades", "camino", "afectadas", "afectan", "barrios", "puerto", "drenajes", "lodo", "falta", "estado", "llevan", "pozo", "servicio", "meses", "alda", "mal", "inundaciones", "san", "exigen", "piden", "santa", "puente", "lluvias", "comunidades", "pobladores", "rio", "familias", "vecinos", "agua"],
        "Eventos Nacionales y Seguridad Pública": ["terror", "autoridades", "muertos", "masacre", "heridas", "juarez", "hallan", "area", "lluvias", "ser", "pacientes", "ataca", "crisis", "veces", "cuerpos", "casos", "volcan", "casas", "mientras", "mujeres", "mes", "hombre", "familias", "casa", "accidente", "tambien", "habria", "balazos", "joven", "horas", "millones", "siguen", "vida", "espera", "carretera", "sido", "afectadas", "muerte", "buscan", "familiares", "ciudad", "migrantes", "mexico", "incendio", "hospital", "blanca", "menos", "arma", "personas", "fuego"],
        "Peticiones y Solicitudes Comunitarias": ["ruta", "barrio", "otros", "personas", "buses", "coatepeque", "sigue", "sur", "santa", "mejoran", "deben", "muro", "caminos", "transito", "policia", "lluvia", "comuna", "zonas", "construccion", "vehicular", "reparan", "accidentes", "estructura", "metros", "ser", "entrega", "caen", "exigen", "temen", "fin", "municipios", "horas", "colonia", "pasaje", "lluvias", "autoridades", "carga", "sector", "principal", "riesgo", "debido", "avenida", "obra", "meses", "zona", "puente", "paso", "transporte", "calle", "vecinos"],
        "Migrantes e Instituciones": ["calle", "mantienen", "final", "reune", "toman", "reporta", "policial", "piden", "75", "ninos", "esperan", "muere", "mp", "alegria", "visita", "500", "madre", "muertes", "general", "an", "tamales", "sectores", "policia", "hechos", "pasado", "muertos", "hijo", "huehuetenango", "violentas", "hogar", "mama", "bebe", "acompana", "ultimos", "fase", "va", "pais", "enero", "personas", "centro", "comerciantes", "presencia", "hoy", "celebran", "primera", "ciudad", "mayor", "vuelta", "mujer", "segunda"],
        "Comunitarios Vial": ["fuerzas", "interamericana", "kilometro", "estado", "temen", "actividades", "mueren", "heridos", "vecinos", "hectareas", "deslizamientos", "meses", "socorristas", "comunitarios", "valle", "usuarios", "bono", "dejan", "policia", "fiestas", "terminal", "delictivos", "tierra", "periferico", "siniestros", "muerto", "fuego", "sector", "victima", "paso", "rio", "hechos", "evita", "pago", "trabajar", "ruta", "prevenir", "departamento", "guatemala", "policias", "forestales", "vigilan", "retiran", "pnc", "piden", "asaltos", "colision", "incendios", "agentes", "seguridad"],
        "Emergencias Nacionales y Seguridad": ["balean", "temperaturas", "familias", "varias", "cada", "familiares", "cabeza", "ser", "ellas", "sufren", "vacaciones", "ofrecer", "joven", "mujer", "celebran", "bloqueos", "cero", "albanil", "visita", "80", "destruye", "veraneantes", "motoristas", "horas", "lleva", "carreteras", "comunidad", "mueren", "600", "santa", "productos", "alza", "menos", "nueva", "corpus", "desconocidos", "ciudad", "nuevo", "asesinado", "christi", "puente", "meses", "esperan", "salud", "cerca", "semana", "precios", "bajo", "aprovechan", "centro"],
        "Desarrollo y Planificación de Infraestructuras": ["atencion", "nacional", "joven", "cierran", "danos", "acceso", "construccion", "recibiran", "transito", "pasar", "pueblo", "punto", "60", "meses", "desaparecido", "regresaba", "santa", "chiquimula", "camino", "autoridades", "jornada", "finalizar", "cristo", "trailer", "estaba", "instalaciones", "muerte", "personas", "casa", "estuvo", "hacia", "fallece", "atropellado", "fin", "esquipulas", "ruta", "semana", "vecinos", "exigen", "carretera", "reparacion", "negro", "pacientes", "alda", "puente", "paso", "arrollado", "hombre", "muere", "hospital"],
        "Delincuencia y Tendencias Criminales": ["clausura", "inician", "bajo", "fundacion", "primer", "cientos", "fallecido", "pan", "guadalupe", "guatemala", "grupo", "recibir", "miles", "fiestas", "hallado", "asuncion", "mundo", "vez", "reciben", "municipio", "semana", "personas", "rosario", "sube", "cadaver", "patrona", "catolicos", "precio", "cantan", "celebra", "mes", "fe", "involucrados", "salud", "dona", "cierra", "carretera", "directo", "aumento", "ruta", "fiesta", "fiestas", "victimas", "guatemala", "piden", "grupos"],
        "Crímenes en el Hogar y Robos": ["accidente", "camion", "unico", "grupo", "victimas", "bloqueo", "ser", "hombres", "muertos", "zona", "municipio", "carreteras", "heridos", "guatemala", "autoridades", "camino", "grande", "defensa", "vuelta", "exigen", "llama", "cooperativa", "tiene", "sufren", "herida", "policiais", "derrumbe", "sismo", "muere", "tragedia", "paso", "familia", "trailer", "veces", "vivo", "accidentes", "rescate", "tiempo", "informe", "delito", "policia", "transito", "traslado", "hombres", "construccion", "ciclo", "hospital", "responsables", "cuidad", "niños", "graves", "nuevas", "carretera"],
        "Clases e Iniciativas Educativas": ["tormenta", "hospital", "afecta", "hombres", "servicio", "estadio", "puente", "ofrecen", "municipios", "personas", "desfiles", "comienza", "itza", "material", "cementerio", "nacional", "san", "torneo", "primera", "potable", "poblacion", "paso", "alerta", "cabecera", "peten", "carretera", "pais", "area", "visitas", "departamento", "covid", "millones", "zonas", "esta", "suspenden", "autoridades", "esperan", "costara", "atencion", "ruta", "rios", "solo", "puesto", "comunidades", "construccion", "agua", "lago", "meses", "centro", "salud"],
        "Emergencias y Respuesta a Crisis": ["chiquimula", "temen", "estaban", "cerca", "detienen", "pasajeros", "chofer", "desconocidos", "agricultor", "aldea", "transito", "llegar", "victimas", "caminaba", "robos", "asaltantes", "bala", "menor", "vecinos", "victima", "salir", "hombres", "joven", "banda", "robo", "ninos", "dejar", "asesinan", "barranco", "eeuu", "mientras", "ayuda", "escapar", "piden", "garcia", "familia", "delincuentes", "vivienda", "zona", "atacan", "muere", "meses", "disparan", "cae", "balazos", "hombre", "vida", "mujer", "atacado", "casa"],
        "Conciencia Pública y Comunicación": ["orfandad", "ninas", "conmemoran", "catolicos", "muerte", "vuelven", "remodelacion", "rafael", "plantel", "estudian", "nuevo", "cientos", "ciudad", "jubilo", "alegria", "casa", "dios", "alumnas", "estudiar", "region", "devocion", "construir", "fondos", "terminar", "nino", "salones", "regresan", "tradicion", "quieren", "hijos", "celebran", "iglesia", "calles", "escuelas", "aldea", "falta", "san", "esta", "exigen", "techo", "reciben", "ninos", "aulas", "familia", "maestros", "alumnos", "padres", "estudiantes", "clases", "escuela"],
        "Seguridad y Regulación del Comercio": ["agresion", "colonia", "balas", "disputa", "local", "hospital", "ver", "asaltos", "sicarios", "pareja", "moto", "camino", "edad", "ayer", "asesinado", "tambien", "disparos", "mientras", "tenia", "asesinan", "acusan", "balazos", "mareros", "sexual", "frente", "robo", "centro", "asesinato", "sector", "pnc", "violacion", "ser", "zona", "matar", "hombres", "barrio", "ladrones", "menor", "supuestos", "capturado", "delincuentes", "extorsion", "disparan", "asalto", "mujer", "victimas", "hombre", "matan", "capturan", "mujeres"],
        "Mejoras y Bienestar Comunitario": ["casco", "canales", "porque", "casi", "hacia", "rutas", "vehiculos", "turismo", "paso", "cuenta", "continuan", "villa", "hay", "pilotos", "invierno", "calle", "afecta", "reparar", "denuncian", "semana", "flores", "mientras", "interamericana", "baches", "personas", "solo", "urbano", "abandono", "kilometros", "comienza", "trabajos", "reparacion", "comercio", "transportistas", "comerciantes", "viven", "accidentes", "exigen", "piden", "san", "festival", "via", "ciudad", "tramo", "autoridades", "visitantes", "turistas", "esperan", "ruta", "vecinos"],
        "Fiestas Nacionales": ["temen", "comuna", "peregrinacion", "nueva", "llego", "nacional", "mes", "municipio", "podrian", "residentes", "encuentran", "pide", "carcel", "mal", "usar", "independencia", "muerte", "espera", "lluvias", "unidos", "personal", "estado", "chimaltenango", "aseguran", "aldea", "recibe", "ingreso", "40", "camino", "solo", "lleva", "hospital", "esperaron", "alternas", "vias", "piden", "edificio", "nuevo", "horas", "muertos", "sido", "meses", "proyecto", "pavimentacion", "abandono", "tiene", "puente", "trabajos", "obra", "vecinos"],
        "Salud Pública y Gestión del Agua": ["calles", "negocio", "mes", "arranca", "hombre", "tierra", "celebran", "olores", "cena", "jorge", "desborde", "rio", "cae", "afecta", "otras", "obra", "comunidades", "esta", "riesgo", "aguas", "metros", "bienvenida", "comuna", "guatemala", "arrancan", "maestro", "afectadas", "150", "nuevo", "policia", "zona", "hipico", "beneficiadas", "antigua", "esperan", "plaza", "jose", "fin", "ninos", "fiestas", "mercado", "ciudad", "drenaje", "actividades", "calle", "vecinos", "familias", "san", "feria", "desfile"]
    }

    # Find a matching category based on top words
    for category, keywords in category_mapping.items():
        if any(word in top_words for word in keywords):
            return category
    
    return "Otros"  # If no match, assign a default category

# Apply category assignment based on the top words of each latent class
data["predicted_category_name_from_words"] = data["latent_class"].apply(
    lambda x: assign_category_to_latent_class(x, topic_keywords)
)

# Print the updated dataframe with predicted category names
print(data[['titulo', 'subtitular', 'latent_class', 'predicted_category_name_from_words']])