import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image

# -----------------------------------------------------------
# CONFIGURATION DE L'APPLICATION
# -----------------------------------------------------------

st.set_page_config(
    page_title="🍎 Fruit Classifier - IA",
    page_icon="🍌",
    layout="centered"
)

st.title("🍉 Fruit Classifier - Détection automatique")
st.write("Glissez-déposez une photo d’un fruit ci-dessous pour découvrir dans quel état est le fruit! 🍊🍏🍒")

# -----------------------------------------------------------
# 📦 CHARGEMENT DU MODÈLE
# -----------------------------------------------------------

@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model("fruit_classifier.keras")
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle : {e}")
        st.error("Assurez-vous que le fichier 'fruit_classifier.keras' se trouve dans le même dossier que votre script app.py")
        return None

model = load_model()

# Liste des classes (DOIT correspondre à l'entraînement)
CLASS_NAMES = ["fresh", "mild", "rotten"] 

IMG_SIZE = (224, 224) # Doit être le même que l'entraînement

# -----------------------------------------------------------
# TÉLÉVERSEMENT DE L’IMAGE
# -----------------------------------------------------------

uploaded_file = st.file_uploader(
    "Glissez une image ici ou cliquez pour en choisir une 📁",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None and model is not None:
    # Affichage de l’image téléchargée
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Image téléchargée", use_container_width=True)

    # Prétraitement (mêmes paramètres que pendant l'entraînement)
    img_resized = image.resize(IMG_SIZE)
    img_array = tf.keras.utils.img_to_array(img_resized)
    img_array = np.expand_dims(img_array, axis=0) # Crée un batch de 1 image

    # -----------------------------------------------------------
    # PRÉDICTION
    # -----------------------------------------------------------

    with st.spinner("Analyse de l’image... 🧠"):
        predictions = model.predict(img_array)
        scores = tf.nn.softmax(predictions[0]).numpy()

    pred_class = CLASS_NAMES[np.argmax(scores)]
    confidence = np.max(scores)

    # -----------------------------------------------------------
    # AFFICHAGE DU RÉSULTAT
    # -----------------------------------------------------------

    st.markdown("---")
    st.subheader("Résultat de la prédiction 🧠")

    # Affichage plus dynamique
    if pred_class == "fresh":
        st.success(f"**État du fruit : {pred_class.capitalize()}** (Confiance : {confidence:.2%}) ")
    elif pred_class == "mild":
        st.warning(f"**État du fruit : {pred_class.capitalize()}** (Confiance : {confidence:.2%}) ")
    else: # rotten
        st.error(f"**État du fruit : {pred_class.capitalize()}** (Confiance : {confidence:.2%}) ")
        
    st.write("Le modèle est confiant à **{:.2f}%** que ce fruit est **{}**.".format(confidence * 100, pred_class))


    # -----------------------------------------------------------
    # VISUALISATION DES PROBABILITÉS (Décommenté et amélioré)
    # -----------------------------------------------------------

    st.subheader("Détail des probabilités")

    # Créer un DataFrame pour un affichage facile
    prob_df = pd.DataFrame({
        "Classe": CLASS_NAMES,
        "Probabilité": scores
    })
    
    # Utiliser st.bar_chart pour un graphique interactif simple
    st.bar_chart(prob_df.set_index("Classe"))

elif model is None:
    st.error("Le modèle n'a pas pu être chargé. L'application ne peut pas fonctionner.")

else:
    st.info("⬆️ Importez une image pour commencer la prédiction.")

# -----------------------------------------------------------
# FOOTER
# -----------------------------------------------------------
st.markdown("---")
st.caption("Application créée par Yann et Téo.")