# ==========================================================================================
# APPLICATION "AUTO KYC" - VERSION PROFESSIONNELLE (UX/UI & BACKEND OPTIMISÉS)
# ==========================================================================================

import streamlit as st
import torch
import cv2
import numpy as np
import json
from PIL import Image
import io
import base64

# L'importation correcte et moderne
from mistralai import Mistral

# --- Importations locales (modèle de détection) ---
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T

# --- CONFIGURATION DU PROJET ---
MODEL_PATH = "frcnn_cni_best_safe.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.8

# --- FONCTIONS DE CHARGEMENT (OPTIMISÉES ET MISES EN CACHE) ---

@st.cache_resource
def load_detection_model():
    """Charge le modèle de détection Faster R-CNN local."""
    try:
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        model.load_state_dict(checkpoint['model_state_dict'])
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

# --- PIPELINE DE TRAITEMENT ---

def detect_cni(model, pil_image):
    """Détecte la CNI et retourne l'image annotée et les coordonnées."""
    image_tensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])(pil_image).to(DEVICE)
    with torch.no_grad():
        prediction = model([image_tensor])
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    best_box = None
    max_score = 0
    for box, score in zip(boxes, scores):
        if score > CONFIDENCE_THRESHOLD and score > max_score:
            max_score = score
            best_box = box
    
    image_np = np.array(pil_image.convert('RGB'))
    image_cv = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    
    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box)
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(image_cv, f"CNI Détectée: {max_score:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    return image_cv, best_box

@st.cache_data(show_spinner=False)
def get_raw_text_from_image(_llm_client, image_bytes):
    """Étape 1 : Appel à l'endpoint OCR spécialisé pour extraire le texte brut."""
    print("Appel à l'API Mistral OCR (non mis en cache)...")
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    data_url = f"data:image/png;base64,{base64_image}"
    try:
        document_source = {"type": "image_url", "image_url": data_url}
        ocr_response = _llm_client.ocr.process(model="mistral-ocr-2505", document=document_source)
        raw_text = "\n".join(page.markdown for page in ocr_response.pages)
        return raw_text
    except Exception as e:
        st.error(f"Erreur lors de l'appel à l'API OCR de Mistral : {e}")
        return None

@st.cache_data(show_spinner=False)
def structure_and_merge_texts(_llm_client, recto_text, verso_text):
    """Étape 2 : Envoie les textes bruts à l'IA de chat pour fusion et structuration."""
    print("Appel à l'API Mistral Chat pour structuration (non mis en cache)...")

    # LE PROMPT "EXPERT RH"
    consolidation_prompt = f"""
    Tu es un agent expert des ressources humaines spécialisé dans la vérification de documents d'identité pour un processus KYC (Know Your Customer) au Cameroun.
    Ta mission est de consolider les informations extraites du RECTO et du VERSO d'une Carte Nationale d'Identité.
    A partir des textes bruts fournis, remplis la fiche d'identité suivante sous forme d'un objet JSON valide.
    Les clés doivent être : "nom", "prenoms", "date_naissance", "lieu_naissance", "sexe", "profession", "pere", "mere", "adresse", "date_delivrance", "date_expiration", "identifiant_unique".
    Sois extrêmement rigoureux. Si une information est manquante ou illisible sur les deux faces, utilise la valeur "Non trouvé".
    Ne fournis AUCUNE explication en dehors de l'objet JSON final.

    Texte extrait du RECTO :
    ---
    {recto_text if recto_text else "Non fourni"}
    ---

    Texte extrait du VERSO :
    ---
    {verso_text if verso_text else "Non fourni"}
    ---
    """
    
    try:
        messages = [{"role": "user", "content": consolidation_prompt}]
        chat_response = _llm_client.chat.complete(
            model="mistral-large-latest",
            messages=messages,
            response_format={"type": "json_object"}
        )
        response_content = chat_response.choices[0].message.content
        structured_data = json.loads(response_content)
        return structured_data
    except Exception as e:
        st.error(f"Erreur lors de la structuration des données par l'IA : {e}")
        return None

# --- COMPOSANTS D'INTERFACE (FRONT-END) ---

def display_identity_card(data):
    """Affiche les informations structurées sous forme de fiche professionnelle."""
    st.subheader("Fiche d'Identité Vérifiée")

    with st.container(border=True):
        st.markdown("##### État Civil")
        col_perso1, col_perso2 = st.columns(2)
        with col_perso1:
            st.text_input("Nom de Famille", value=data.get('nom', 'N/A'), disabled=True)
            st.text_input("Prénoms", value=data.get('prenoms', 'N/A'), disabled=True)
            st.text_input("Profession", value=data.get('profession', 'N/A'), disabled=True)
        with col_perso2:
            st.text_input("Date de Naissance", value=data.get('date_naissance', 'N/A'), disabled=True)
            st.text_input("Lieu de Naissance", value=data.get('lieu_naissance', 'N/A'), disabled=True)
            st.text_input("Sexe", value=data.get('sexe', 'N/A'), disabled=True)
        
        st.divider()
        
        st.markdown("##### Filiation")
        col_fil1, col_fil2 = st.columns(2)
        with col_fil1:
            st.text_input("Père", value=data.get('pere', 'N/A'), disabled=True)
        with col_fil2:
            st.text_input("Mère", value=data.get('mere', 'N/A'), disabled=True)
        
        st.divider()

        st.markdown("##### Informations du Document")
        st.text_input("Adresse Enregistrée", value=data.get('adresse', 'N/A'), disabled=True)
        col_doc1, col_doc2, col_doc3 = st.columns(3)
        with col_doc1:
            st.text_input("Date de Délivrance", value=data.get('date_delivrance', 'N/A'), disabled=True)
        with col_doc2:
            st.text_input("Date d'Expiration", value=data.get('date_expiration', 'N/A'), disabled=True)
        with col_doc3:
            st.text_input("Identifiant Unique", value=data.get('identifiant_unique', 'N/A'), disabled=True)

# --- APPLICATION PRINCIPALE STREAMLIT ---

def main():
    st.set_page_config(page_title="Auto KYC", layout="wide", initial_sidebar_state="collapsed")
    
    # --- EN-TÊTE ---
    st.title("🆔 Auto KYC")
    st.markdown("**Optimisez vos processus de vérification d'identité.** Chargez le recto et le verso d'une CNI pour une extraction de données automatique, rapide et fiable.")
    st.info("ℹ️ Les images chargées sont traitées de manière sécurisée et ne sont pas conservées après la session.", icon="🛡️")
    st.divider()

    # --- LAYOUT PRINCIPAL ---
    col_actions, col_resultats = st.columns([2, 3]) # Colonne d'actions plus petite

    with col_actions:
        st.subheader("1. Charger les Documents")
        recto_file = st.file_uploader("Chargez le RECTO", type=["jpg", "jpeg", "png"])
        verso_file = st.file_uploader("Chargez le VERSO", type=["jpg", "jpeg", "png"])

        process_button = st.button("Lancer la Vérification KYC ✨", type="primary", use_container_width=True)

    with col_resultats:
        st.subheader("2. Résultats de la Vérification")

        if process_button:
            if not recto_file and not verso_file:
                st.warning("Veuillez charger au moins une face de la carte d'identité.")
                st.stop()

            # Initialisation des placeholders pour les résultats
            if 'results' not in st.session_state:
                st.session_state.results = {}
            
            detection_model = load_detection_model()
            llm_client = load_llm_client()

            if not (detection_model and llm_client):
                st.error("Un des modèles n'a pas pu être chargé. L'analyse est interrompue.")
                st.stop()
            
            recto_text, verso_text = None, None
            
            # Traitement dynamique du recto et du verso
            with st.status("Analyse en cours...", expanded=True) as status:
                if recto_file:
                    st.write("Analyse du recto...")
                    pil_image = Image.open(recto_file)
                    detected_image, box = detect_cni(detection_model, pil_image)
                    if box is not None:
                        x1, y1, x2, y2 = map(int, box)
                        crop = pil_image.crop((x1, y1, x2, y2))
                        with io.BytesIO() as buf:
                            crop.save(buf, format='PNG')
                            image_bytes = buf.getvalue()
                        recto_text = get_raw_text_from_image(llm_client, image_bytes)
                    st.session_state.results['recto_img'] = detected_image
                
                if verso_file:
                    st.write("Analyse du verso...")
                    pil_image = Image.open(verso_file)
                    detected_image, box = detect_cni(detection_model, pil_image)
                    if box is not None:
                        x1, y1, x2, y2 = map(int, box)
                        crop = pil_image.crop((x1, y1, x2, y2))
                        with io.BytesIO() as buf:
                            crop.save(buf, format='PNG')
                            image_bytes = buf.getvalue()
                        verso_text = get_raw_text_from_image(llm_client, image_bytes)
                    st.session_state.results['verso_img'] = detected_image

                if recto_text or verso_text:
                    st.write("Consolidation des données par l'IA...")
                    structured_data = structure_and_merge_texts(llm_client, recto_text, verso_text)
                    st.session_state.results['data'] = structured_data
                    st.session_state.results['raw_recto'] = recto_text
                    st.session_state.results['raw_verso'] = verso_text
                
                status.update(label="Analyse terminée !", state="complete", expanded=False)

        # Affichage des résultats en dehors du bloc "if process_button" pour la persistance
        if 'results' in st.session_state and st.session_state.results:
            
            res_col1, res_col2 = st.columns(2)
            if 'recto_img' in st.session_state.results:
                with res_col1:
                    st.image(st.session_state.results['recto_img'], caption="Recto analysé", channels="BGR", use_container_width=True)
            if 'verso_img' in st.session_state.results:
                with res_col2:
                    st.image(st.session_state.results['verso_img'], caption="Verso analysé", channels="BGR", use_container_width=True)

            if 'data' in st.session_state.results and st.session_state.results['data']:
                display_identity_card(st.session_state.results['data'])
                
                with st.expander("Voir les textes bruts extraits pour le débogage"):
                    st.text_area("Texte Brut du Recto", value=st.session_state.results.get('raw_recto') or "Non traité", height=150)
                    st.text_area("Texte Brut du Verso", value=st.session_state.results.get('raw_verso') or "Non traité", height=150)
            else:
                st.error("L'extraction et la structuration des données ont échoué.")
        else:
            st.info("Les résultats de la vérification apparaîtront ici.")

if __name__ == "__main__":
    main()