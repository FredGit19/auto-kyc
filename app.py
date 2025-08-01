# ==========================================================================================
# APPLICATION "AUTO KYC" - v3.1 - CORRECTION DÉFINITIVE DU TYPEERROR
# ==========================================================================================
# Cause Racine de l'Erreur Précédente : L'objet `map` en Python 3 est un itérateur
# non-subscriptable. La fonction `crop` de PIL attend une séquence (tuple/liste).
# Solution : Forcer l'évaluation de l'itérateur en le convertissant explicitement
# en tuple via `tuple(map(int, box))`.
# Statut : Cette version corrige la dernière erreur d'exécution connue.
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

# --- PIPELINE DE TRAITEMENT MODULAIRE ET BLINDÉ ---

def preprocess_uploaded_file(uploaded_file):
    if not uploaded_file: return None
    try:
        image_bytes = uploaded_file.getvalue()
        image = Image.open(io.BytesIO(image_bytes))
        image = ImageOps.exif_transpose(image.convert("RGB"))
        if image.width > MAX_IMAGE_DIMENSION or image.height > MAX_IMAGE_DIMENSION:
            image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)
        return image
    except Exception as e:
        st.warning(f"Impossible de traiter le fichier '{uploaded_file.name}'. Il est peut-être corrompu. Erreur: {e}")
        return None

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
    cv2.putText(image_cv, f"CNI Détectée ({best_score:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (34, 139, 34), 2)
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
        st.error(f"Erreur API OCR Mistral: {e}")
        return None

@st.cache_data(show_spinner=False)
def generate_kyc_report(_llm_client, recto_text, verso_text):
    if not _llm_client or (not recto_text and not verso_text): return None
    prompt = f"""
    En tant qu'agent de vérification KYC, analyse les textes extraits d'une CNI camerounaise.
    Produis un objet JSON valide sans aucune explication.
    Le JSON doit contenir "analyse_coherence" et "donnees_extraites".
    1. "analyse_coherence":
        - "score_coherence_textuelle": Entier (0-100) basé sur la complétude, validité des formats de date (JJ/MM/AAAA), et absence de bruit OCR.
        - "points_verification": Liste de chaînes décrivant les points vérifiés.
    2. "donnees_extraites":
        - Extrais: "nom", "prenoms", "date_naissance", "lieu_naissance", "sexe", "profession", "pere", "mere", "adresse", "date_delivrance", "date_expiration", "identifiant_unique".
        - Utilise "Non trouvé" si l'info est absente.
    RECTO: --- {recto_text or "Non fourni"} ---
    VERSO: --- {verso_text or "Non fourni"} ---
    """
    try:
        messages = [{"role": "user", "content": prompt}]
        response = _llm_client.chat.complete(model="mistral-large-latest", messages=messages, response_format={"type": "json_object"})
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        st.error(f"Erreur API Chat Mistral: {e}")
        return None

def process_single_side(side_name, uploaded_file, detection_model, llm_client):
    st.write(f"**Analyse du {side_name}...**")
    pil_image = preprocess_uploaded_file(uploaded_file)
    if not pil_image:
        st.warning(f"Le fichier {side_name} n'a pas pu être pré-traité.")
        return None

    annotated_img, box = detect_cni(detection_model, pil_image)
    if box is None:
        st.warning(f"Aucune CNI détectée sur le {side_name}.")
        return {"annotated_img": annotated_img, "text": None}

    try:
        # <<< LA CORRECTION DÉFINITIVE EST ICI >>>
        # On convertit l'itérateur map en tuple avant de le passer à crop.
        crop_box = tuple(map(int, box))
        crop = pil_image.crop(crop_box)
        
        with io.BytesIO() as buf:
            crop.save(buf, format='PNG')
            image_bytes = buf.getvalue()
        
        text = get_text_from_image_via_ocr(llm_client, image_bytes)
        if text:
            st.success(f"Texte extrait du {side_name}.")
        else:
            st.warning(f"L'OCR n'a retourné aucun texte pour le {side_name}.")

        return {"annotated_img": annotated_img, "text": text}
    except Exception as e:
        st.error(f"Erreur inattendue lors du rognage ou de l'OCR du {side_name}: {e}")
        return {"annotated_img": annotated_img, "text": None}

def display_results_ui(report, recto_result, verso_result):
    st.subheader("2. Résultats de la Vérification")
    if not report:
        st.error("Le rapport final n'a pas pu être généré.")
        return

    auth_data = report.get("analyse_coherence")
    id_data = report.get("donnees_extraites")

    if auth_data:
        with st.container(border=True):
            st.markdown("##### Analyse de Cohérence Textuelle")
            score = auth_data.get('score_coherence_textuelle', 0)
            label = "Élevée" if score >= 85 else "Moyenne" if score >= 60 else "Faible"
            st.progress(score, f"Score: {score}/100 - Cohérence {label}")
            with st.expander("Détails de la vérification"):
                for point in auth_data.get('points_verification', []): st.markdown(f"- {point}")
    
    if id_data:
        with st.container(border=True):
            st.markdown("##### Fiche d'Identité Extraite")
            for key, value in id_data.items():
                st.text_input(key.replace("_", " ").title(), value, disabled=True, key=f"id_{key}")

    st.markdown("##### Images Analysées")
    col1, col2 = st.columns(2)
    if recto_result: col1.image(recto_result["annotated_img"], caption="Recto traité", channels="BGR")
    if verso_result: col2.image(verso_result["annotated_img"], caption="Verso traité", channels="BGR")

def main():
    st.set_page_config(page_title="Auto KYC", layout="wide")
    st.title("🆔 Auto KYC")
    st.markdown("Système de Vérification et d'Extraction pour CNI. Conçu pour la fiabilité.")
    st.divider()

    detection_model = load_detection_model()
    llm_client = load_llm_client()

    if not detection_model or not llm_client:
        st.error("Un composant critique est manquant. L'application ne peut pas démarrer.")
        st.stop()

    col_actions, col_results = st.columns([2, 3])
    with col_actions:
        st.subheader("1. Charger les Documents")
        recto_file = st.file_uploader("Chargez le RECTO (Image/PDF)", type=["jpg", "jpeg", "png"])
        verso_file = st.file_uploader("Chargez le VERSO (Image/PDF)", type=["jpg", "jpeg", "png"])

        if st.button("Lancer la Vérification ✨", type="primary", use_container_width=True):
            if not recto_file and not verso_file:
                st.warning("Veuillez charger au moins une face de la carte.")
                st.stop()
            
            with st.status("Traitement du document...", expanded=True) as status:
                recto_result = process_single_side("Recto", recto_file, detection_model, llm_client) if recto_file else None
                verso_result = process_single_side("Verso", verso_file, detection_model, llm_client) if verso_file else None
                
                recto_text = recto_result["text"] if recto_result else None
                verso_text = verso_result["text"] if verso_result else None

                report = None
                if recto_text or verso_text:
                    status.update(label="Analyse par l'IA...")
                    report = generate_kyc_report(llm_client, recto_text, verso_text)
                
                st.session_state.report = report
                st.session_state.recto_result = recto_result
                st.session_state.verso_result = verso_result
                status.update(label="Analyse terminée !", state="complete")

    with col_results:
        if 'report' in st.session_state:
            display_results_ui(st.session_state.get('report'), st.session_state.get('recto_result'), st.session_state.get('verso_result'))
        else:
            st.info("Les résultats de la vérification apparaîtront ici.")

if __name__ == "__main__":
     main() 