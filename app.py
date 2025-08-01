# ==========================================================================================
# APPLICATION "AUTO KYC" - ARCHITECTURE FINALE ET ROBUSTE v2.0
# ==========================================================================================
# Expert Review & Refactoring:
# - Performance: Handles large files via pre-processing (resizing, PDF image extraction).
# - Speed: Leverages GPU for detection, smart caching.
# - Prompting: Precision-engineered prompt for reliable, unbiased data extraction and textual coherence scoring.
# - UX: Non-blocking feel, clear status updates, and factual reporting without subjective AI comments.
# ==========================================================================================
import base64
import streamlit as st
import torch
import cv2
import numpy as np
import json
from PIL import Image
import io
import fitz  
import PyMuPDF
import os
import time

# --- Importations locales ---
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T
# Assurez-vous d'avoir une API Key Mistral dans vos secrets
try:
    from mistralai import Mistral
except ImportError:
    st.error("La bibliothèque Mistral AI n'est pas installée. Veuillez exécuter `pip install mistralai`.")
    st.stop()

# --- CONFIGURATION DU PROJET ---
MODEL_PATH = "frcnn_cni_best_safe.pth"
# Utiliser le GPU si disponible, c'est une optimisation majeure de la vitesse.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.8
# Limite de taille d'image pour le traitement afin d'éviter les crashs mémoire et les temps de traitement excessifs.
MAX_IMAGE_DIMENSION = 1280

# --- FONCTIONS DE CHARGEMENT (Mise en cache) ---

@st.cache_resource
def load_detection_model():
    """Charge le modèle de détection Faster R-CNN."""
    try:
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
        # Il est crucial que le modèle ait été entraîné et sauvegardé correctement.
        # Ici, on suppose un .pth contenant uniquement le state_dict.
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        model.to(DEVICE)
        model.eval()
        return model
    except FileNotFoundError:
        st.error(f"FATAL: Le fichier du modèle '{MODEL_PATH}' est introuvable. L'application ne peut pas démarrer.")
        return None
    except Exception as e:
        st.error(f"Erreur critique au chargement du modèle de détection: {e}")
        return None

@st.cache_resource
def load_llm_client():
    """Initialise le client Mistral AI à partir des secrets Streamlit."""
    if "MISTRAL_API_KEY" not in st.secrets:
        st.error("FATAL: La clé MISTRAL_API_KEY n'est pas configurée dans les secrets de Streamlit.")
        return None
    try:
        client = Mistral(api_key=st.secrets["MISTRAL_API_KEY"])
        return client
    except Exception as e:
        st.error(f"Erreur d'initialisation du client Mistral: {e}")
        return None

# --- PIPELINE DE TRAITEMENT OPTIMISÉ ---

def preprocess_uploaded_file(uploaded_file):
    """
    Gère les PDF et les images lourdes.
    Extrait la première image d'un PDF ou redimensionne une image trop grande.
    """
    if uploaded_file is None:
        return None

    file_bytes = uploaded_file.getvalue()
    
    if uploaded_file.type == "application/pdf":
        try:
            pdf_document = fitz.open(stream=file_bytes, filetype="pdf")
            if pdf_document.page_count == 0:
                st.warning(f"Le fichier PDF '{uploaded_file.name}' est vide ou corrompu.")
                return None
            first_page = pdf_document.load_page(0)
            pix = first_page.get_pixmap(dpi=200) # Augmenter le DPI pour une meilleure qualité
            image = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        except Exception as e:
            st.error(f"Erreur lors de la lecture du PDF '{uploaded_file.name}': {e}")
            return None
    else:
        image = Image.open(io.BytesIO(file_bytes)).convert("RGB")

    # Redimensionnement systématique pour la performance
    if image.width > MAX_IMAGE_DIMENSION or image.height > MAX_IMAGE_DIMENSION:
        image.thumbnail((MAX_IMAGE_DIMENSION, MAX_IMAGE_DIMENSION), Image.Resampling.LANCZOS)
        
    return image

def detect_cni(model, pil_image):
    """Détecte la CNI sur une image PIL, retourne une image annotée (OpenCV) et les coordonnées."""
    image_tensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])(pil_image).to(DEVICE)
    with torch.no_grad():
        prediction = model([image_tensor])
        
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    
    best_box = None
    max_score = 0
    if len(boxes) > 0:
        best_idx = np.argmax(scores)
        if scores[best_idx] > CONFIDENCE_THRESHOLD:
            max_score = scores[best_idx]
            best_box = boxes[best_idx]

    image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box)
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"CNI Détectée: {max_score:.2f}"
        cv2.putText(image_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
    return image_cv, best_box

@st.cache_data(show_spinner=False)
# --- PIPELINE DE TRAITEMENT OPTIMISÉ (SECTION CORRIGÉE) ---

@st.cache_data(show_spinner=False)
def get_text_from_cni_crop(_llm_client, image_bytes):
    """
    Appel unique à l'OCR de Mistral, utilisant le format d'appel API correct.
    """
    try:
        # Étape 1: Encoder l'image en base64, le format requis par l'API pour les data URLs.
        base64_image = base64.b64encode(image_bytes).decode('utf-8')
        
        # Étape 2: Créer l'objet 'document' conforme à la spécification de l'API.
        document_source = {
            "type": "image_url",
            "image_url": f"data:image/png;base64,{base64_image}"
        }
        
        # Étape 3: Appeler la méthode avec le bon nom d'argument ('document').
        response = _llm_client.ocr.process(
            model="mistral-ocr-2505", 
            document=document_source
        )
        
        return "\n".join(page.markdown for page in response.pages)
    
    except Exception as e:
        # Fournir un message d'erreur plus utile pour le débogage.
        st.error(f"Erreur lors de l'appel à l'API OCR de Mistral. Vérifiez votre clé API et les données envoyées. Détails: {e}")
        return None

@st.cache_data(show_spinner=False)
def perform_kyc_analysis(_llm_client, recto_text, verso_text):
    """
    PROMPT DE PRÉCISION V2.0 :
    - Objectif clair: Validation de cohérence textuelle et extraction de données.
    - Contraintes fortes: Format JSON strict, pas de baratin.
    - Logique de scoring définie: Le LLM est guidé sur *comment* calculer le score.
    """
    if not recto_text and not verso_text:
        return None

    expert_prompt = f"""
    En tant qu'agent de vérification KYC, analyse les textes extraits d'une CNI camerounaise.
    Ta mission est d'évaluer la cohérence TEXTUELLE et d'extraire les données.
    Produis un objet JSON valide sans aucune explication préliminaire.

    Le JSON doit contenir deux clés de haut niveau: "analyse_coherence" et "donnees_extraites".

    1.  Pour "analyse_coherence":
        - "score_coherence_textuelle": Un entier de 0 à 100. Calcule ce score en te basant sur:
            - La présence de champs clés (nom, date de naissance, identifiant).
            - La validité du format des dates (JJ/MM/AAAA).
            - L'absence de caractères visiblement erronés ou de "bruit OCR".
            - La cohérence entre la date de délivrance et d'expiration.
        - "points_verification": Une liste de chaînes de caractères décrivant les points positifs et négatifs observés (ex: "Format de la date de naissance valide.", "Bruit OCR détecté dans la section 'adresse'.").

    2.  Pour "donnees_extraites":
        - Extrais les informations dans les clés suivantes: "nom", "prenoms", "date_naissance", "lieu_naissance", "sexe", "profession", "pere", "mere", "adresse", "date_delivrance", "date_expiration", "identifiant_unique".
        - Si une information n'est pas présente dans les textes fournis, utilise la chaîne de caractères "Non trouvé".

    Texte du RECTO:
    ---
    {recto_text or "Non fourni"}
    ---

    Texte du VERSO:
    ---
    {verso_text or "Non fourni"}
    ---
    """
    try:
        messages = [{"role": "user", "content": expert_prompt}]
        chat_response = _llm_client.chat.complete(
            model="mistral-large-latest", messages=messages, response_format={"type": "json_object"}
        )
        return json.loads(chat_response.choices[0].message.content)
    except Exception as e:
        st.error(f"Erreur API Chat Mistral: {e}")
        return None

# --- COMPOSANTS D'INTERFACE (UI) ---

def display_results(report):
    """Affiche le rapport final de manière structurée."""
    st.subheader("2. Résultats de la Vérification")
    
    auth_data = report.get("analyse_coherence")
    id_data = report.get("donnees_extraites")

    if not auth_data or not id_data:
        st.error("Le rapport généré par l'IA est incomplet ou malformé.")
        return

    # Section 1: Analyse de Cohérence
    with st.container(border=True):
        st.markdown("##### Analyse de Cohérence Textuelle")
        score = auth_data.get('score_coherence_textuelle', 0)
        
        if score > 85: color, label = "green", "Cohérence Élevée"
        elif score > 60: color, label = "orange", "Cohérence Moyenne"
        else: color, label = "red", "Cohérence Faible"
        
        st.progress(score, text=f"Score: {score}/100 - {label}")
        
        with st.expander("Détails de la vérification"):
            for point in auth_data.get('points_verification', []):
                st.markdown(f"- {point}")

    # Section 2: Fiche d'Identité
    with st.container(border=True):
        st.markdown("##### Fiche d'Identité Extraite")
        for key, value in id_data.items():
            st.text_input(label=key.replace("_", " ").title(), value=value, disabled=True, key=f"id_{key}")

    # Section 3: Images traitées
    st.markdown("##### Images Analysées")
    res_col1, res_col2 = st.columns(2)
    if 'recto_img' in st.session_state:
        res_col1.image(st.session_state.recto_img, caption="Recto traité", channels="BGR", use_column_width=True)
    if 'verso_img' in st.session_state:
        res_col2.image(st.session_state.verso_img, caption="Verso traité", channels="BGR", use_column_width=True)

# --- APPLICATION PRINCIPALE ---

def main():
    st.set_page_config(page_title="Auto KYC", layout="wide", initial_sidebar_state="collapsed")
    st.title("🆔 Auto KYC")
    st.markdown("Système expert pour la **Vérification de Cohérence** et l'**Extraction de Données** des CNI.")
    st.divider()

    # Charger les modèles une seule fois au démarrage
    detection_model = load_detection_model()
    llm_client = load_llm_client()
    if not detection_model or not llm_client:
        st.warning("Un ou plusieurs modèles n'ont pas pu être chargés. L'application est en mode dégradé.")
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
                # Nettoyer l'état de la session précédente
                for key in list(st.session_state.keys()):
                    if key not in ['detection_model', 'llm_client']: # Ne pas effacer les modèles cachés
                        del st.session_state[key]
                
                with st.spinner("Analyse en cours..."):
                    recto_text, verso_text = None, None
                    
                    pil_recto = preprocess_uploaded_file(recto_file)
                    if pil_recto:
                        detected_img, box = detect_cni(detection_model, pil_recto)
                        st.session_state.recto_img = detected_img
                        if box is not None:
                            x1, y1, x2, y2 = map(int, box)
                            crop = pil_recto.crop((x1, y1, x2, y2))
                            with io.BytesIO() as buf:
                                crop.save(buf, format='PNG')
                                recto_text = get_text_from_cni_crop(llm_client, buf.getvalue())

                    pil_verso = preprocess_uploaded_file(verso_file)
                    if pil_verso:
                        detected_img, box = detect_cni(detection_model, pil_verso)
                        st.session_state.verso_img = detected_img
                        if box is not None:
                            x1, y1, x2, y2 = map(int, box)
                            crop = pil_verso.crop((x1, y1, x2, y2))
                            with io.BytesIO() as buf:
                                crop.save(buf, format='PNG')
                                verso_text = get_text_from_cni_crop(llm_client, buf.getvalue())

                    if recto_text or verso_text:
                        final_report = perform_kyc_analysis(llm_client, recto_text, verso_text)
                        st.session_state.final_report = final_report

    with col_resultats:
        if 'final_report' in st.session_state and st.session_state.final_report:
            display_results(st.session_state.final_report)
        else:
            st.info("Les résultats de la vérification apparaîtront ici.")

if __name__ == "__main__":
    main()