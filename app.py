# ==========================================================================================
# APPLICATION "AUTO KYC" - VERSION DE PRODUCTION DÉFINITIVE
# ==========================================================================================

import streamlit as st
import torch
import cv2
import numpy as np
import json
from PIL import Image
import io
import base64
import os

from mistralai.client import Mistral

# --- Importations locales (modèle de détection) ---
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T

# --- CONFIGURATION DU PROJET ---
MODEL_PATH = "frcnn_cni_best_safe.pth"
DEVICE = torch.device("cpu")
CONFIDENCE_THRESHOLD = 0.8

# --- FONCTIONS DE CHARGEMENT ---

@st.cache_resource
def load_detection_model():
    """Charge le modèle de détection depuis le fichier local."""
    try:
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
        
        # <<< LA CORRECTION DÉFINITIVE EST ICI >>>
        # Le fichier .pth est directement le state_dict, pas un checkpoint.
        # On le charge donc directement dans le modèle.
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        
        model.to(DEVICE)
        model.eval()
        print("Modèle de détection chargé.")
        return model
    except Exception as e:
        st.error(f"Erreur au chargement du modèle de détection: {e}")
        return None

@st.cache_resource
def load_llm_client():
    """Initialise le client Mistral AI."""
    try:
        api_key = st.secrets["MISTRAL_API_KEY"]
        client = Mistral(api_key=api_key)
        print("Client Mistral AI initialisé.")
        return client
    except Exception as e:
        st.error(f"Erreur d'initialisation du client Mistral. Avez-vous configuré .streamlit/secrets.toml ? Erreur: {e}")
        return None

# --- PIPELINE DE TRAITEMENT (INCHANGÉ) ---
# ... (Toutes les autres fonctions : detect_cni, get_raw_text_from_image, structure_and_merge_texts, authenticate_cni_image, etc. sont parfaites) ...
def detect_cni(model, pil_image):
    image_tensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])(pil_image).to(DEVICE)
    with torch.no_grad():
        prediction = model([image_tensor])
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    best_box = None; max_score = 0
    for box, score in zip(boxes, scores):
        if score > CONFIDENCE_THRESHOLD and score > max_score:
            max_score = score
            best_box = box
    image_np = np.array(pil_image.convert('RGB')); image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box)
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(image_cv, f"CNI Détectée: {max_score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image_cv, best_box

@st.cache_data(show_spinner=False)
def get_raw_text_from_image(_llm_client, image_bytes):
    print("Appel à l'API Mistral OCR (non mis en cache)...")
    base64_image = base64.b64encode(image_bytes).decode('utf-8'); data_url = f"data:image/png;base64,{base64_image}"
    try:
        document_source = {"type": "image_url", "image_url": data_url}
        ocr_response = _llm_client.ocr.process(model="mistral-ocr-2505", document=document_source)
        return "\n".join(page.markdown for page in ocr_response.pages)
    except Exception as e: st.error(f"Erreur lors de l'appel à l'API OCR de Mistral : {e}"); return None

@st.cache_data(show_spinner=False)
def structure_and_merge_texts(_llm_client, recto_text, verso_text):
    print("Appel à l'API Mistral Chat pour structuration (non mis en cache)...")
    consolidation_prompt = f"""
    Tu es un agent expert RH. Consolide les informations des textes RECTO et VERSO d'une CNI du Cameroun en un objet JSON valide.
    Les clés doivent être : "nom", "prenoms", "date_naissance", "lieu_naissance", "sexe", "profession", "pere", "mere", "adresse", "date_delivrance", "date_expiration", "identifiant_unique".
    Si une info est manquante, utilise "Non trouvé". Ne fournis AUCUNE explication en dehors du JSON.
    Texte RECTO : --- {recto_text if recto_text else "Non fourni"} ---
    Texte VERSO : --- {verso_text if verso_text else "Non fourni"} ---"""
    try:
        messages = [{"role": "user", "content": consolidation_prompt}]
        chat_response = _llm_client.chat.complete(model="mistral-large-latest", messages=messages, response_format={"type": "json_object"})
        return json.loads(chat_response.choices[0].message.content)
    except Exception as e: st.error(f"Erreur lors de la structuration des données par l'IA : {e}"); return None

@st.cache_data(show_spinner=False)
def authenticate_cni_image(_llm_client, image_bytes):
    print("Appel à l'API Mistral Chat pour authentification...")
    base64_image = base64.b64encode(image_bytes).decode('utf-8'); data_url = f"data:image/png;base64,{base64_image}"
    auth_prompt = """
    Tu es un expert en documents forensiques. Analyse l'image de CNI fournie.
    Vérifie l'intégrité de la photo, la cohérence de la typographie, les éléments de sécurité, et la qualité générale.
    Retourne ton analyse UNIQUEMENT sous forme d'un objet JSON valide avec les clés :
    "score_plausibilite" (0-100), "niveau_risque" ("Faible", "Moyen", "Élevé"), "observations" (une liste), et "recommandation"."""
    try:
        messages = [{"role": "user", "content": [{"type": "text", "text": auth_prompt}, {"type": "image_url", "image_url": {"url": data_url}}]}]
        chat_response = _llm_client.chat.complete(model="mistral-large-latest", messages=messages, response_format={"type": "json_object"})
        return json.loads(chat_response.choices[0].message.content)
    except Exception as e: st.error(f"Erreur lors de l'analyse d'authenticité par l'IA : {e}"); return None

def display_authentication_report(auth_data):
    st.subheader("Rapport d'Authentification")
    with st.container(border=True):
        score = auth_data.get('score_plausibilite', 0)
        if score > 85: color = "green"; delta = "Risque Faible"
        elif score > 60: color = "orange"; delta = "Risque Moyen"
        else: color = "red"; delta = "Risque Élevé"
        st.metric(label="Score de Plausibilité", value=f"{score} / 100", delta=delta, delta_color="inverse")
        st.info(f"**Recommandation de l'IA :** {auth_data.get('recommandation', 'N/A')}", icon="💡")
        with st.expander("Voir les observations détaillées"):
            for obs in auth_data.get('observations', []): st.markdown(f"- {obs}")

def display_identity_card(data):
    st.subheader("Fiche d'Identité Extraite")
    with st.container(border=True):
        st.markdown("##### État Civil"); col1, col2 = st.columns(2)
        with col1:
            st.text_input("Nom", value=data.get('nom', 'N/A'), disabled=True)
            st.text_input("Prénoms", value=data.get('prenoms', 'N/A'), disabled=True)
            st.text_input("Profession", value=data.get('profession', 'N/A'), disabled=True)
        with col2:
            st.text_input("Date de Naissance", value=data.get('date_naissance', 'N/A'), disabled=True)
            st.text_input("Lieu de Naissance", value=data.get('lieu_naissance', 'N/A'), disabled=True)
            st.text_input("Sexe", value=data.get('sexe', 'N/A'), disabled=True)
        st.divider(); st.markdown("##### Filiation"); col3, col4 = st.columns(2)
        with col3: st.text_input("Père", value=data.get('pere', 'N/A'), disabled=True)
        with col4: st.text_input("Mère", value=data.get('mere', 'N/A'), disabled=True)
        st.divider(); st.markdown("##### Informations du Document")
        st.text_input("Adresse", value=data.get('adresse', 'N/A'), disabled=True)
        col5, col6, col7 = st.columns(3)
        with col5: st.text_input("Délivrance", value=data.get('date_delivrance', 'N/A'), disabled=True)
        with col6: st.text_input("Expiration", value=data.get('date_expiration', 'N/A'), disabled=True)
        with col7: st.text_input("Identifiant Unique", value=data.get('identifiant_unique', 'N/A'), disabled=True)

def main():
    st.set_page_config(page_title="Auto KYC", layout="wide", initial_sidebar_state="collapsed")
    st.title("🆔 Auto KYC"); st.markdown("Le système expert pour l'**Authentification** et l'**Extraction de Données** des CNI.")
    st.info("ℹ️ Les images sont traitées de manière sécurisée et ne sont pas conservées.", icon="🛡️"); st.divider()
    col_actions, col_resultats = st.columns([2, 3])
    with col_actions:
        st.subheader("1. Charger les Documents")
        recto_file = st.file_uploader("Chargez le RECTO", type=["jpg", "jpeg", "png"])
        verso_file = st.file_uploader("Chargez le VERSO", type=["jpg", "jpeg", "png"])
        process_button = st.button("Lancer la Vérification KYC ✨", type="primary", use_container_width=True)
    with col_resultats:
        st.subheader("2. Résultats de la Vérification")
        if process_button:
            if not recto_file and not verso_file:
                st.warning("Veuillez charger au moins une face de la carte."); st.stop()
            st.session_state.clear()
            detection_model, llm_client = load_detection_model(), load_llm_client()
            if not (detection_model and llm_client): st.stop()
            recto_text, verso_text, auth_recto, auth_verso = None, None, None, None
            with st.status("Analyse en cours...", expanded=True) as status:
                if recto_file:
                    status.update(label="Analyse du recto..."); pil_image = Image.open(recto_file)
                    detected_image, box = detect_cni(detection_model, pil_image)
                    st.session_state.recto_img = detected_image
                    if box is not None:
                        x1, y1, x2, y2 = map(int, box); crop = pil_image.crop((x1, y1, x2, y2))
                        with io.BytesIO() as buf: crop.save(buf, format='PNG'); image_bytes = buf.getvalue()
                        auth_recto = authenticate_cni_image(llm_client, image_bytes)
                        recto_text = get_raw_text_from_image(llm_client, image_bytes)
                if verso_file:
                    status.update(label="Analyse du verso..."); pil_image = Image.open(verso_file)
                    detected_image, box = detect_cni(detection_model, pil_image)
                    st.session_state.verso_img = detected_image
                    if box is not None:
                        x1, y1, x2, y2 = map(int, box); crop = pil_image.crop((x1, y1, x2, y2))
                        with io.BytesIO() as buf: crop.save(buf, format='PNG'); image_bytes = buf.getvalue()
                        auth_verso = authenticate_cni_image(llm_client, image_bytes)
                        verso_text = get_raw_text_from_image(llm_client, image_bytes)
                if recto_text or verso_text:
                    status.update(label="Consolidation des données...")
                    st.session_state.data = structure_and_merge_texts(llm_client, recto_text, verso_text)
                st.session_state.auth_recto = auth_recto; st.session_state.auth_verso = auth_verso
                status.update(label="Analyse terminée !", state="complete", expanded=False)
        if 'data' in st.session_state and st.session_state.data:
            display_identity_card(st.session_state.data)
            auth_col1, auth_col2 = st.columns(2)
            if 'auth_recto' in st.session_state and st.session_state.auth_recto:
                with auth_col1:
                    st.image(st.session_state.recto_img, caption="Recto analysé", channels="BGR", use_container_width=True)
                    display_authentication_report(st.session_state.auth_recto)
            if 'auth_verso' in st.session_state and st.session_state.auth_verso:
                with auth_col2:
                    st.image(st.session_state.verso_img, caption="Verso analysé", channels="BGR", use_container_width=True)
                    display_authentication_report(st.session_state.auth_verso)
        else: st.info("Les résultats de la vérification apparaîtront ici.")

if __name__ == "__main__":
    main()