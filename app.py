# ==========================================================================================
# APPLICATION "AUTO KYC" - v2.1 - VERSION STABILISÉE ET CORRIGÉE
# ==========================================================================================
# Révisions Clés :
# - CORRECTION : Résolution de l'erreur d'import de `fitz` en assumant un `requirements.txt` correct (PyMuPDF).
# - CORRECTION : Utilisation de la signature d'API OCR Mistral correcte (`document` au lieu de `files`).
# - CONSOLIDATION : Intégration de toutes les optimisations de performance et de prompting précédentes.
# - ROBUSTESSE : Amélioration de la gestion des erreurs et des messages à l'utilisateur.
# ==========================================================================================

import streamlit as st
import torch
import cv2
import numpy as np
import json
from PIL import Image
import io
import fitz  # PyMuPDF
import base64

# --- Importations locales et gestion des dépendances ---
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
try:
    from mistralai import Mistral
except ImportError:
    st.error("Dépendance manquante : La bibliothèque Mistral AI n'est pas installée. Veuillez l'ajouter à votre requirements.txt (`mistralai`).")
    st.stop()

# --- CONFIGURATION DU PROJET ---
MODEL_PATH = "frcnn_cni_best_safe.pth"
# Utiliser le GPU si disponible pour une accélération significative
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.8
# Limite de taille pour normaliser les images et garantir la performance
MAX_IMAGE_DIMENSION = 1280

# --- FONCTIONS DE CHARGEMENT (Mise en cache pour la performance) ---

@st.cache_resource
def load_detection_model():
    """Charge le modèle de détection Faster R-CNN local."""
    if not os.path.exists(MODEL_PATH):
        st.error(f"FATAL: Fichier modèle introuvable à l'emplacement '{MODEL_PATH}'. L'application ne peut pas fonctionner.")
        return None
    try:
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except Exception as e:
        st.error(f"Erreur critique au chargement du modèle de détection: {e}")
        return None

@st.cache_resource
def load_llm_client():
    """Initialise le client Mistral AI à partir des secrets Streamlit."""
    if "MISTRAL_API_KEY" not in st.secrets:
        st.error("FATAL: La clé MISTRAL_API_KEY n'est pas configurée dans les secrets Streamlit (.streamlit/secrets.toml).")
        return None
    try:
        return Mistral(api_key=st.secrets["MISTRAL_API_KEY"])
    except Exception as e:
        st.error(f"Erreur d'initialisation du client Mistral: {e}")
        return None

# --- PIPELINE DE TRAITEMENT OPTIMISÉ ---

def preprocess_uploaded_file(uploaded_file):
    """
    Gère les PDF et les images lourdes en les normalisant.
    Extrait la première image d'un PDF ou redimensionne une image trop grande.
    """
    if uploaded_file is None: return None
    file_bytes = uploaded_file.getvalue()
    try:
        if uploaded_file.type == "application/pdf":
            pdf_doc = fitz.open(stream=file_bytes, filetype="pdf")
            if not pdf_doc.page_count: return None # PDF vide
            page = pdf_doc.load_page(0)
            pix = page.get_pixmap(dpi=200)
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        else:
            image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

        if image.width > MAX_IMAGE_DIMENSION or image.height > MAX_IMAGE_DIMENSION:
            image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)
        return image
    except Exception as e:
        st.warning(f"Impossible de traiter le fichier '{uploaded_file.name}'. Il est peut-être corrompu. Erreur: {e}")
        return None

def detect_cni(model, pil_image):
    """Détecte la CNI, retourne une image annotée (format OpenCV) et les coordonnées du cadre."""
    image_tensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])(pil_image).to(DEVICE)
    with torch.no_grad():
        prediction = model([image_tensor])
    boxes, scores = prediction[0]['boxes'].cpu().numpy(), prediction[0]['scores'].cpu().numpy()
    best_box = None
    if len(boxes) > 0:
        best_idx = np.argmax(scores)
        if scores[best_idx] > CONFIDENCE_THRESHOLD:
            best_box = boxes[best_idx]

    image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box)
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(image_cv, f"CNI Détectée: {scores[best_idx]:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image_cv, best_box

@st.cache_data(show_spinner=False)
def get_text_from_cni_crop(_llm_client, image_bytes):
    """Appel à l'API OCR de Mistral avec la signature d'API correcte."""
    try:
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        document_source = {"type": "image_url", "image_url": f"data:image/png;base64,{base64_image}"}
        response = _llm_client.ocr.process(model="mistral-ocr-2505", document=document_source)
        return "\n".join(page.markdown for page in response.pages)
    except Exception as e:
        st.error(f"Erreur API OCR Mistral: {e}")
        return None

@st.cache_data(show_spinner=False)
def perform_kyc_analysis(_llm_client, recto_text, verso_text):
    """Exécute le prompt de précision v2.0 pour l'analyse et l'extraction."""
    if not recto_text and not verso_text: return None
    expert_prompt = f"""
    En tant qu'agent de vérification KYC, analyse les textes extraits d'une CNI camerounaise.
    Produis un objet JSON valide sans aucune explication.
    Le JSON doit contenir deux clés: "analyse_coherence" et "donnees_extraites".
    1. Pour "analyse_coherence":
        - "score_coherence_textuelle": Un entier de 0 à 100 basé sur la complétude des champs clés, la validité du format des dates (JJ/MM/AAAA), et l'absence de bruit OCR.
        - "points_verification": Une liste de chaînes de caractères décrivant les points positifs et négatifs observés (ex: "Format de la date de naissance valide.", "Bruit OCR détecté dans l'adresse.").
    2. Pour "donnees_extraites":
        - Extrais les informations dans les clés suivantes: "nom", "prenoms", "date_naissance", "lieu_naissance", "sexe", "profession", "pere", "mere", "adresse", "date_delivrance", "date_expiration", "identifiant_unique".
        - Utilise "Non trouvé" si une information est manquante.
    Texte du RECTO: --- {recto_text or "Non fourni"} ---
    Texte du VERSO: --- {verso_text or "Non fourni"} ---
    """
    try:
        messages = [{"role": "user", "content": expert_prompt}]
        chat_response = _llm_client.chat.complete(model="mistral-large-latest", messages=messages, response_format={"type": "json_object"})
        return json.loads(chat_response.choices[0].message.content)
    except Exception as e:
        st.error(f"Erreur API Chat Mistral: {e}")
        return None

# --- COMPOSANTS D'INTERFACE (UI) ---

def display_results(report):
    st.subheader("2. Résultats de la Vérification")
    auth_data = report.get("analyse_coherence")
    id_data = report.get("donnees_extraites")
    if not auth_data or not id_data:
        st.error("Rapport IA incomplet. Impossible d'afficher les résultats.")
        return

    with st.container(border=True):
        st.markdown("##### Analyse de Cohérence Textuelle")
        score = auth_data.get('score_coherence_textuelle', 0)
        label = "Élevée" if score > 85 else "Moyenne" if score > 60 else "Faible"
        st.progress(score, text=f"Score: {score}/100 - Cohérence {label}")
        with st.expander("Détails de la vérification"):
            for point in auth_data.get('points_verification', []): st.markdown(f"- {point}")

    with st.container(border=True):
        st.markdown("##### Fiche d'Identité Extraite")
        for key, value in id_data.items():
            st.text_input(label=key.replace("_", " ").title(), value=value, disabled=True, key=f"id_{key}")

    st.markdown("##### Images Analysées")
    col1, col2 = st.columns(2)
    if 'recto_img' in st.session_state: col1.image(st.session_state.recto_img, caption="Recto traité", channels="BGR", use_column_width=True)
    if 'verso_img' in st.session_state: col2.image(st.session_state.verso_img, caption="Verso traité", channels="BGR", use_column_width=True)

# --- APPLICATION PRINCIPALE ---

def main():
    st.set_page_config(page_title="Auto KYC", layout="wide", initial_sidebar_state="collapsed")
    st.title("🆔 Auto KYC")
    st.markdown("Système expert pour la **Vérification de Cohérence** et l'**Extraction de Données** des CNI.")
    st.divider()

    detection_model = load_detection_model()
    llm_client = load_llm_client()
    if not detection_model or not llm_client:
        st.error("Un ou plusieurs modèles critiques n'ont pas pu être chargés. L'application ne peut pas continuer.")
        st.stop()

    col_actions, col_resultats = st.columns([2, 3])
    with col_actions:
        st.subheader("1. Charger les Documents")
        recto_file = st.file_uploader("Chargez le RECTO (Image ou PDF)", type=["jpg", "jpeg", "png", "pdf"])
        verso_file = st.file_uploader("Chargez le VERSO (Image ou PDF)", type=["jpg", "jpeg", "png", "pdf"])
        if st.button("Lancer la Vérification ✨", type="primary", use_container_width=True):
            if not recto_file and not verso_file:
                st.warning("Veuillez charger au moins une face de la carte.")
            else:
                for key in list(st.session_state.keys()):
                    if key not in ['detection_model', 'llm_client']: del st.session_state[key]
                with st.spinner("Analyse en cours..."):
                    recto_text, verso_text = None, None
                    pil_recto = preprocess_uploaded_file(recto_file)
                    if pil_recto:
                        detected_img, box = detect_cni(detection_model, pil_recto)
                        st.session_state.recto_img = detected_img
                        if box is not None:
                            crop = pil_recto.crop(map(int, box))
                            with io.BytesIO() as buf: crop.save(buf, format='PNG'); recto_text = get_text_from_cni_crop(llm_client, buf.getvalue())
                    
                    pil_verso = preprocess_uploaded_file(verso_file)
                    if pil_verso:
                        detected_img, box = detect_cni(detection_model, pil_verso)
                        st.session_state.verso_img = detected_img
                        if box is not None:
                            crop = pil_verso.crop(map(int, box))
                            with io.BytesIO() as buf: crop.save(buf, format='PNG'); verso_text = get_text_from_cni_crop(llm_client, buf.getvalue())

                    if recto_text or verso_text:
                        st.session_state.final_report = perform_kyc_analysis(llm_client, recto_text, verso_text)

    with col_resultats:
        if 'final_report' in st.session_state and st.session_state.final_report:
            display_results(st.session_state.final_report)
        else:
            st.info("Les résultats de la vérification apparaîtront ici.")

if __name__ == "__main__":
    # Ajout d'une vérification des importations critiques avant de lancer main()
    import os
    if not os.path.exists('requirements.txt'):
        st.error("Le fichier 'requirements.txt' est manquant. Il est essentiel pour le déploiement.")
    main()