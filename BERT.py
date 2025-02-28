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
import torch

df = pd.read_csv("Datos_Portadas.csv")
data = pd.DataFrame(df)
print(data.describe())

spanish_stopwords = [
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las",
    "por", "un", "para", "con", "no", "una", "su", "al", "es", "lo",
    "como", "más", "pero", "sus", "le", "ya", "o", "fue", "me", "si",
    "sin", "sobre", "este", "ya", "también", "entre", "cuando", "uno", "dos",
    "tres", "cuatro", "cinco", "seis", "siete", "ocho", "nueve", "mas", "diez", "1",
    "2", "3", "4", "5", "6", "7", "8", "9", "10", "11","12", "13", "14", "15", "16",
    "17", "18", "19", "20", "21", "22", "23", "24", "25", "26", "27", "28", "29", "30",
    "aã", "tras", "mil", "anos", "tras", "niã", "son", "contra", "fueron", "hasta",
    "queda", "hasta", "os", "sera", "durante", "van", "han", "q1", "q2", "q3", "q4", "q5", "q6",
    "q7", "q8", "q9", "q10", "deja", "ante", "han", "estan", "pierde", "ha", "dia", "50", "2022",
    "desde", "despues", "ano", "dias", "otra", "luego", "km", "pedro", "2024", "2025", "2023", "hace",
    "donde", "otro", "daã", "iba", "les", "dan", "45", "tienen", "hacen", "juan"
]

vectorizer = CountVectorizer(max_features=5000, stop_words=spanish_stopwords)

data["titulo"] = data["titulo"].fillna("").astype(str)
data["subtitular"] = data["subtitular"].fillna("").astype(str)

data["text"] = data["titulo"] + " " + data["subtitular"]


n_components_range = [5, 10, 15, 20, 25]  
num_classes_range = [5, 10, 15, 20]  


distributions = []

for n_components in n_components_range:
    for num_classes in num_classes_range:

        vectorizer = CountVectorizer(max_features=5000, stop_words=spanish_stopwords)
        X = vectorizer.fit_transform(data["text"])

        lda = LatentDirichletAllocation(n_components=n_components, random_state=42)
        X_topics = lda.fit_transform(X)
        data["topic"] = X_topics.argmax(axis=1)

        encoder = OneHotEncoder(sparse_output=False)
        X_lca = encoder.fit_transform(data[["topic"]])

        lca_model = GaussianMixture(n_components=num_classes, covariance_type="full", random_state=42)
        data["latent_class"] = lca_model.fit_predict(X_lca)

        class_distribution = data["latent_class"].value_counts().values
        distributions.append([n_components, num_classes, class_distribution])

distributions_df = pd.DataFrame(distributions, columns=["n_components", "num_classes", "class_distribution"])

distributions_df["class_count"] = distributions_df["class_distribution"].apply(lambda x: np.sum(x))

scaler = StandardScaler()
X_lca_scaled = scaler.fit_transform(X_lca)

num_classes = 19

lca_model = GaussianMixture(n_components=num_classes, covariance_type="spherical", init_params="kmeans", random_state=20)
data["latent_class"] = lca_model.fit_predict(X_lca)

print(data["latent_class"].value_counts())

new_title = "taxis pirata estan en la mira"
new_subtitle = "autoridades buscan diezmar inseguridad y criminalidad"

new_text = new_title + " " + new_subtitle

X_new = vectorizer.transform([new_text])
X_new_topic = lda.transform(X_new).argmax(axis=1)  

X_new_lca = encoder.transform([[X_new_topic[0]]])

predicted_class = lca_model.predict(X_new_lca)
print("Predicted Latent Class:", predicted_class[0])

beto_model_name = "dccuchile/bert-base-spanish-wwm-cased"
tokenizer = AutoTokenizer.from_pretrained(beto_model_name)
model = AutoModel.from_pretrained(beto_model_name)

def get_beto_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    
    with torch.no_grad():
        outputs = model(**inputs)

    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()

data["beto_embedding"] = data.apply(lambda row: get_beto_embedding(row["titulo"] + " " + row["subtitular"]), axis=1)

X_beto = np.vstack(data["beto_embedding"].values)

scaler = StandardScaler()
X_beto_scaled = scaler.fit_transform(X_beto)

# Define number of latent classes (use same num_classes as before)
num_classes = 19

# Fit LCA model with BETO embeddings
lca_model = GaussianMixture(n_components=num_classes, covariance_type="spherical", init_params="kmeans", random_state=20)
data["latent_class"] = lca_model.fit_predict(X_beto_scaled)

# Check class distribution
print(data["latent_class"].value_counts())

# New article
new_title = "Taxis pirata están en la mira"
new_subtitle = "Autoridades buscan diezmar inseguridad y criminalidad"
new_text = new_title + " " + new_subtitle

# Get BETO embedding
X_new_beto = get_beto_embedding(new_text).reshape(1, -1)

# Standardize
X_new_scaled = scaler.transform(X_new_beto)

# Predict latent class
predicted_class = lca_model.predict(X_new_scaled)
print("Predicted Latent Class:", predicted_class[0])

# Save updated DataFrame with assigned latent classes
data.to_csv("Datos_Portadas_Categorizados.csv", index=False)

print("Updated CSV file saved as 'Datos_Portadas_Categorizados.csv'")
