# ==========================================================================================
# APPLICATION "AUTO KYC" - v4.0 - VERSION "ARBITRAGE"
# ==========================================================================================
# Objectif : Fournir un verdict binaire ("Conforme" / "Vérification Manuelle") et une fiche d'identité.
# Logique d'Authentification :
# 1. Détection Visuelle > Seuil de confiance élevé.
# 2. Analyse par LLM pour vérifier la complétude des champs critiques et la cohérence des données.
# 3. Le LLM retourne un verdict de conformité textuelle qui, combiné à la détection, donne le résultat final.
# UI : Interface épurée, sans score ni détails techniques, pour une expérience utilisateur directe.
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
# Seuil de détection visuelle plus strict pour la conformité
DETECTION_THRESHOLD_CONFORMITY = 0.90 
MAX_IMAGE_DIMENSION = 1280

# --- INITIALISATION DES RESSOURCES ---

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

# --- PIPELINE DE TRAITEMENT ---

def preprocess_uploaded_file(uploaded_file):
    if not uploaded_file: return None
    try:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image.convert("RGB"))
        if image.width > MAX_IMAGE_DIMENSION or image.height > MAX_IMAGE_DIMENSION:
            image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)
        return image
    except Exception:
        return None

def detect_cni(model, pil_image):
    if not model or not pil_image: return None, 0.0
    image_tensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])(pil_image).to(DEVICE)
    with torch.no_grad():
        prediction = model([image_tensor])
    boxes, scores = prediction[0]['boxes'].cpu().numpy(), prediction[0]['scores'].cpu().numpy()
    if len(scores) == 0: return None, 0.0
    best_idx = np.argmax(scores)
    return boxes[best_idx], scores[best_idx]

@st.cache_data(show_spinner=False)
def get_text_from_image_via_ocr(_llm_client, image_bytes):
    if not _llm_client or not image_bytes: return None
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        document = {"type": "image_url", "image_url": f"data:image/png;base64,{base64_image}"}
        response = _llm_client.ocr.process(model="mistral-ocr-2505", document=document)
        return "\n".join(page.markdown for page in response.pages)
    except Exception:
        return None

@st.cache_data(show_spinner=False)
def perform_kyc_analysis(_llm_client, recto_text, verso_text):
    """
    PROMPT D'ARBITRAGE v4.0 :
    - Objectif : Extraire les données ET retourner un verdict de conformité textuelle.
    """
    if not _llm_client or (not recto_text and not verso_text): return None
    prompt = f"""
    Tu es un agent de vérification KYC spécialisé dans les CNI Camerounaises.
    Ta mission est d'analyser le texte extrait et de retourner un objet JSON valide sans aucune explication.
    Le JSON doit contenir deux clés : "conformite_textuelle" (un booléen) et "donnees_extraites".

    1. Pour déterminer "conformite_textuelle" (true/false):
       - Le document est NON CONFORME si l'un des champs critiques suivants est manquant : 'nom', 'prenoms', 'date_naissance', 'identifiant_unique'.
       - Le document est NON CONFORME si le format des dates est invalide.
       - Le document est NON CONFORME s'il y a un bruit OCR excessif.
       - Sinon, le document est CONFORME.

    2. Pour "donnees_extraites":
       - Extrais les informations suivantes : "nom", "prenoms", "date_naissance", "lieu_naissance", "sexe", "profession", "pere", "mere", "adresse", "date_delivrance", "date_expiration", "identifiant_unique".
       - Utilise "Non trouvé" si une information est absente.

    RECTO: --- {recto_text or "Non fourni"} ---
    VERSO: --- {verso_text or "Non fourni"} ---
    """
    try:
        messages = [{"role": "user", "content": prompt}]
        response = _llm_client.chat.complete(model="mistral-large-latest", messages=messages, response_format={"type": "json_object"})
        return json.loads(response.choices[0].message.content)
    except Exception:
        return None

def process_single_side(side_name, uploaded_file, detection_model, llm_client):
    pil_image = preprocess_uploaded_file(uploaded_file)
    if not pil_image: return None

    box, score = detect_cni(detection_model, pil_image)
    if box is None:
        return {"conformity_check": False, "reason": f"Aucune CNI détectée sur le {side_name}.", "data": None}

    if score < DETECTION_THRESHOLD_CONFORMITY:
        return {"conformity_check": False, "reason": f"Détection de CNI sur le {side_name} jugée peu fiable (score: {score:.2f}).", "data": None}

    try:
        crop_box = tuple(map(int, box))
        crop = pil_image.crop(crop_box)
        with io.BytesIO() as buf:
            crop.save(buf, format='PNG')
            image_bytes = buf.getvalue()
        
        text = get_text_from_image_via_ocr(llm_client, image_bytes)
        if not text:
            return {"conformity_check": False, "reason": f"La lecture du texte (OCR) a échoué sur le {side_name}.", "data": None}
        
        return {"conformity_check": True, "reason": "Détection visuelle et lecture réussies.", "data": text}
    except Exception:
        return {"conformity_check": False, "reason": f"Erreur technique lors du traitement du {side_name}.", "data": None}

# --- INTERFACE UTILISATEUR (UI) ---

def display_identity_card(data):
    with st.container(border=True):
        st.markdown("##### Fiche d'Identité")
        for key, value in data.items():
            st.text_input(key.replace("_", " ").title(), value, disabled=True, key=f"id_{key}")

def main():
    st.set_page_config(page_title="Auto KYC", layout="wide")
    st.title("🆔 Système de Vérification d'Identité")
    st.markdown("Chargez les deux faces de la carte nationale d'identité pour lancer la vérification.")
    st.divider()

    models = {'detection': load_detection_model(), 'llm': load_llm_client()}
    if not models['detection'] or not models['llm']:
        st.error("Un composant critique est manquant. L'application ne peut pas démarrer.")
        st.stop()

    col_actions, col_results = st.columns([2, 3])
    with col_actions:
        st.subheader("1. Charger les documents")
        recto_file = st.file_uploader("Chargez le RECTO (Image/PDF)", type=["jpg", "jpeg", "png"])
        verso_file = st.file_uploader("Chargez le VERSO (Image/PDF)", type=["jpg", "jpeg", "png"])

        if st.button("Lancer la Vérification ✨", type="primary", use_container_width=True):
            if not recto_file or not verso_file:
                st.warning("Veuillez charger les DEUX faces de la carte.")
                st.stop()
            
            with st.spinner("Analyse en cours..."):
                recto_result = process_single_side("Recto", recto_file, models['detection'], models['llm'])
                verso_result = process_single_side("Verso", verso_file, models['detection'], models['llm'])

                # Logique de décision finale
                is_visually_conform = recto_result and recto_result["conformity_check"] and verso_result and verso_result["conformity_check"]
                
                final_report = None
                if is_visually_conform:
                    final_report = perform_kyc_analysis(models['llm'], recto_result["data"], verso_result["data"])
                
                # Sauvegarder les résultats pour l'affichage
                st.session_state.final_report = final_report
                st.session_state.recto_reason = recto_result["reason"] if recto_result else "Fichier non traité."
                st.session_state.verso_reason = verso_result["reason"] if verso_result else "Fichier non traité."

    with col_results:
        st.subheader("2. Résultat de la vérification")
        if 'final_report' in st.session_state:
            report = st.session_state.final_report
            
            # La décision finale combine la conformité visuelle et textuelle
            is_textually_conform = report and report.get("conformite_textuelle", False)
            
            if is_textually_conform:
                st.success("✅ CNI Présente et Conforme", icon="✅")
                id_data = report.get("donnees_extraites")
                if id_data:
                    display_identity_card(id_data)
            else:
                st.error("⚠️ Vérification Manuelle Requise", icon="⚠️")
                with st.expander("Détails de l'analyse"):
                    st.markdown(f"**Recto :** {st.session_state.recto_reason}")
                    st.markdown(f"**Verso :** {st.session_state.verso_reason}")
                    if report and not report.get("conformite_textuelle"):
                        st.markdown("**Analyse IA :** La structure textuelle extraite a été jugée non conforme.")
        else:
            st.info("Les résultats de la vérification apparaîtront ici.")

if __name__ == "__main__":
    main()