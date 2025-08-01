# ==========================================================================================
# APPLICATION "AUTO KYC" - v4.0 - VERSION "VERDICT & EXTRACTION"
# ==========================================================================================
# Flux de Travail Simplifié pour l'Opérateur :
# 1. Chargement des documents.
# 2. L'IA effectue une analyse de conformité en arrière-plan.
# 3. Affichage d'un verdict clair ("Conforme" / "Vérification Manuelle Requise").
# 4. Affichage des données extraites si le document est jugé conforme.
# ==========================================================================================

import streamlit as st
import torch
import cv2
import numpy as np
import json
from PIL import Image, ImageOps
import io
import fitz  # PyMuPDF
import base64
import os

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
try:
    from mistralai import Mistral
except ImportError:
    st.error("Dépendance manquante : `mistralai`. Veuillez l'ajouter à votre requirements.txt.")
    st.stop()

# --- CONFIGURATION GLOBALE ---
MODEL_PATH = "frcnn_cni_best_safe.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.8
MAX_IMAGE_DIMENSION = 1280
# Seuil de score de cohérence pour déclarer un document "Conforme"
CONFORMITY_THRESHOLD_SCORE = 75

# --- INITIALISATION DES RESSOURCES (Mise en cache) ---

@st.cache_resource
def load_detection_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"FATAL: Fichier modèle introuvable '{MODEL_PATH}'.")
        return None
    try:
        model = fasterrcnn_resnet50_fpn(weights=None)
        model.roi_heads.box_predictor = FastRCNNPredictor(model.roi_heads.box_predictor.cls_score.in_features, 2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE).eval()
        return model
    except Exception as e:
        st.error(f"Erreur critique au chargement du modèle: {e}")
        return None

@st.cache_resource
def load_llm_client():
    if "MISTRAL_API_KEY" not in st.secrets:
        st.error("FATAL: MISTRAL_API_KEY non configurée dans les secrets Streamlit.")
        return None
    try:
        return Mistral(api_key=st.secrets["MISTRAL_API_KEY"])
    except Exception as e:
        st.error(f"Erreur d'initialisation du client Mistral: {e}")
        return None

# --- PIPELINE DE TRAITEMENT MODULAIRE ---

def preprocess_uploaded_file(uploaded_file):
    if not uploaded_file: return None
    try:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image.convert("RGB"))
        if image.width > MAX_IMAGE_DIMENSION or image.height > MAX_IMAGE_DIMENSION:
            image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)
        return image
    except Exception: return None

def detect_cni(model, pil_image):
    if not model or not pil_image: return None, None
    image_tensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])(pil_image).to(DEVICE)
    with torch.no_grad():
        prediction = model([image_tensor])
    boxes, scores = prediction[0]['boxes'].cpu().numpy(), prediction[0]['scores'].cpu().numpy()
    if len(scores) == 0: return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR), None
    best_idx = np.argmax(scores)
    best_score = scores[best_idx]
    if best_score < CONFIDENCE_THRESHOLD: return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR), None
    best_box = boxes[best_idx]
    image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    x1, y1, x2, y2 = map(int, best_box)
    cv2.rectangle(image_cv, (x1, y1), (x2, y2), (34, 139, 34), 3)
    return image_cv, best_box

@st.cache_data(show_spinner=False)
def get_text_from_image_via_ocr(_llm_client, image_bytes):
    if not _llm_client or not image_bytes: return None
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        document = {"type": "image_url", "image_url": f"data:image/png;base64,{base64_image}"}
        response = _llm_client.ocr.process(model="mistral-ocr-2505", document=document)
        return "\n".join(page.markdown for page in response.pages)
    except Exception as e:
        st.error(f"Erreur API OCR: {e}")
        return None

@st.cache_data(show_spinner=False)
def get_kyc_verdict_and_data(_llm_client, recto_text, verso_text):
    """
    PROMPT DE VALIDATION V4.0 :
    Demande un verdict binaire et les données extraites, basés sur une analyse de cohérence interne.
    """
    if not _llm_client or (not recto_text and not verso_text): return None
    
    # Ce prompt est le nouveau "cerveau" de l'application.
    validation_prompt = f"""
    En tant qu'agent de validation KYC expert pour le Cameroun, ta seule mission est d'analyser les textes OCR d'une CNI.
    Produis un objet JSON unique et valide, sans aucune autre explication.

    Le JSON doit contenir deux clés de haut niveau : "verdict" et "donnees_extraites".

    1.  Pour la clé "verdict", analyse la cohérence, la complétude et la plausibilité des textes fournis.
        - Si les champs clés (nom, date de naissance, identifiant) sont présents, les formats de date sont corrects (JJ/MM/AAAA), et il y a peu de bruit OCR, la valeur doit être "CONFORME".
        - Dans tous les autres cas (champs manquants, formats de date invalides, texte incohérent ou très bruité), la valeur doit être "VÉRIFICATION MANUELLE REQUISE".
        - Base ton jugement uniquement sur le texte fourni.

    2.  Pour la clé "donnees_extraites", extrais les informations de manière structurée.
        - Les clés doivent être : "nom", "prenoms", "date_naissance", "lieu_naissance", "sexe", "profession", "pere", "mere", "adresse", "date_delivrance", "date_expiration", "identifiant_unique".
        - Si une information n'est pas trouvée, utilise la chaîne "Non trouvé".

    Texte du RECTO :
    ---
    {recto_text or "Non fourni"}
    ---

    Texte du VERSO :
    ---
    {verso_text or "Non fourni"}
    ---
    """
    try:
        messages = [{"role": "user", "content": validation_prompt}]
        response = _llm_client.chat.complete(
            model="mistral-large-latest", messages=messages, response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Erreur API Chat: {e}")
        return None

def process_single_side(side_name, uploaded_file, detection_model):
    """Encapsule la détection pour une face. Retourne l'image annotée et l'image rognée."""
    pil_image = preprocess_uploaded_file(uploaded_file)
    if not pil_image: return None, None
    annotated_img, box = detect_cni(detection_model, pil_image)
    if box is None: return annotated_img, None
    try:
        crop_box = tuple(map(int, box))
        crop_img = pil_image.crop(crop_box)
        return annotated_img, crop_img
    except Exception: return annotated_img, None

# --- INTERFACE UTILISATEUR (UI) ---

def display_verdict(verdict_text):
    if verdict_text == "CONFORME":
        st.success("✅ CNI Présente et Conforme", icon="✔")
    else:
        st.warning("⚠️ Vérification Manuelle Requise", icon="❗")

def display_identity_card(data):
    st.subheader("Fiche d'Identité Extraite")
    with st.container(border=True):
        for key, value in data.items():
            st.text_input(key.replace("_", " ").title(), value, disabled=True, key=f"id_{key}")

def main():
    st.set_page_config(page_title="Auto KYC", layout="wide")
    st.title("🆔 Outil de Vérification de CNI")
    st.markdown("Chargez les deux faces d'une carte d'identité pour lancer la vérification.")
    st.divider()

    detection_model = load_detection_model()
    llm_client = load_llm_client()
    if not detection_model or not llm_client:
        st.stop()

    col_actions, col_results = st.columns([2, 3])
    with col_actions:
        st.subheader("1. Charger les Documents")
        recto_file = st.file_uploader("Chargez le RECTO (Image)", type=["jpg", "jpeg", "png"])
        verso_file = st.file_uploader("Chargez le VERSO (Image)", type=["jpg", "jpeg", "png"])

        if st.button("Lancer la Vérification ✨", type="primary", use_container_width=True):
            if not recto_file or not verso_file:
                st.warning("Veuillez charger les deux faces de la carte.")
                st.stop()
            
            with st.spinner("Analyse en cours..."):
                annotated_recto, crop_recto = process_single_side("Recto", recto_file, detection_model)
                annotated_verso, crop_verso = process_single_side("Verso", verso_file, detection_model)
                
                st.session_state.annotated_recto = annotated_recto
                st.session_state.annotated_verso = annotated_verso

                recto_text, verso_text = None, None
                if crop_recto:
                    with io.BytesIO() as buf: crop_recto.save(buf, format='PNG'); recto_text = get_text_from_image_via_ocr(llm_client, buf.getvalue())
                if crop_verso:
                    with io.BytesIO() as buf: crop_verso.save(buf, format='PNG'); verso_text = get_text_from_image_via_ocr(llm_client, buf.getvalue())
                
                report = None
                if recto_text or verso_text:
                    report = get_kyc_verdict_and_data(llm_client, recto_text, verso_text)
                
                st.session_state.report = report

    with col_results:
        st.subheader("2. Résultats")
        if 'report' in st.session_state and st.session_state.report:
            report = st.session_state.report
            verdict = report.get("verdict", "VÉRIFICATION MANUELLE REQUISE")
            id_data = report.get("donnees_extraites")

            display_verdict(verdict)

            if id_data:
                display_identity_card(id_data)

            st.markdown("##### Documents Analysés")
            res_col1, res_col2 = st.columns(2)
            if 'annotated_recto' in st.session_state: res_col1.image(st.session_state.annotated_recto, caption="Recto", channels="BGR")
            if 'annotated_verso' in st.session_state: res_col2.image(st.session_state.annotated_verso, caption="Verso", channels="BGR")
        
        elif 'annotated_recto' in st.session_state or 'annotated_verso' in st.session_state:
             # Gère le cas où la détection a fonctionné mais l'OCR/LLM a échoué
            st.warning("⚠️ Vérification Manuelle Requise", icon="❗")
            st.markdown("La détection a fonctionné, mais les données n'ont pas pu être extraites ou validées.")
            st.markdown("##### Documents Analysés")
            res_col1, res_col2 = st.columns(2)
            if 'annotated_recto' in st.session_state: res_col1.image(st.session_state.annotated_recto, caption="Recto", channels="BGR")
            if 'annotated_verso' in st.session_state: res_col2.image(st.session_state.annotated_verso, caption="Verso", channels="BGR")
        else:
            st.info("Les résultats de la vérification apparaîtront ici.")

if __name__ == "__main__":
    main()