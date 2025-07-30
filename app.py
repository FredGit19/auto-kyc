# ==========================================================================================
# APPLICATION "AUTO KYC" - VERSION EXPERT POUR SECTEUR BANCAIRE
# Architecture haute performance et analyse intelligente
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
import tempfile
import pyvips  # Essentiel pour la performance sur fichiers volumineux

from mistralai import Mistral

# --- Importations locales pour le modèle de détection ---
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import v2 as T

# --- CONFIGURATION STRATÉGIQUE ---
MODEL_PATH = "frcnn_cni_best_safe.pth"
DEVICE = torch.device("cpu")
CONFIDENCE_THRESHOLD = 0.8
TILE_SIZE = 1280
TILE_OVERLAP = 100

# --- CHARGEMENT OPTIMISÉ DES RESSOURCES ---
@st.cache_resource
def load_detection_model():
    """Charge le modèle de détection de CNI."""
    try:
        model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        print("Modèle de détection chargé.")
        return model
    except Exception as e:
        st.error(f"Erreur critique au chargement du modèle de détection : {e}")
        return None

@st.cache_resource
def load_llm_client():
    """Initialise le client Mistral AI."""
    try:
        client = Mistral(api_key=st.secrets["MISTRAL_API_KEY"])
        print("Client Mistral AI initialisé.")
        return client
    except Exception as e:
        st.error(f"Erreur de configuration du client Mistral. Vérifiez .streamlit/secrets.toml. Détails : {e}")
        return None

# --- PIPELINE DE TRAITEMENT D'IMAGE "OUT-OF-CORE" ---
def detect_on_tile(model, tile_pil):
    """Exécute la détection sur une seule tuile (format PIL)."""
    image_tensor = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])(tile_pil).to(DEVICE)
    with torch.no_grad():
        prediction = model([image_tensor])
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    best_box = None; max_score = 0
    for box, score in zip(boxes, scores):
        if score > CONFIDENCE_THRESHOLD and score > max_score:
            max_score = score
            best_box = box
    return best_box

def process_large_image_in_tiles(model, image_path):
    """Scanne une image volumineuse par tuiles sans la charger en mémoire."""
    try:
        vips_image = pyvips.Image.new_from_file(image_path, access='sequential')
        width, height = vips_image.width, vips_image.height
        thumbnail_vips = vips_image.thumbnail_image(800)
        thumbnail_np = np.ndarray(
            buffer=thumbnail_vips.write_to_memory(),
            dtype=np.uint8, 
            shape=[thumbnail_vips.height, thumbnail_vips.width, thumbnail_vips.bands]
        )
        if thumbnail_np.shape[2] == 4: thumbnail_np = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGBA2BGR)
        else: thumbnail_np = cv2.cvtColor(thumbnail_np, cv2.COLOR_RGB2BGR)

        for y in range(0, height, TILE_SIZE - TILE_OVERLAP):
            for x in range(0, width, TILE_SIZE - TILE_OVERLAP):
                tile_width, tile_height = min(TILE_SIZE, width - x), min(TILE_SIZE, height - y)
                if tile_width <= 0 or tile_height <= 0: continue
                tile_vips = vips_image.crop(x, y, tile_width, tile_height)
                tile_np = np.ndarray(buffer=tile_vips.write_to_memory(), dtype=np.uint8, shape=[tile_height, tile_width, tile_vips.bands])
                tile_pil = Image.fromarray(tile_np)
                box = detect_on_tile(model, tile_pil)
                if box is not None:
                    global_box = (x + box[0], y + box[1], x + box[2], y + box[3])
                    scale_x, scale_y = thumbnail_np.shape[1] / width, thumbnail_np.shape[0] / height
                    cv2.rectangle(thumbnail_np, 
                                  (int(global_box[0] * scale_x), int(global_box[1] * scale_y)), 
                                  (int(global_box[2] * scale_x), int(global_box[3] * scale_y)), 
                                  (0, 255, 0), 3)
                    return global_box, thumbnail_np
        return None, thumbnail_np
    except pyvips.error.VipsError as e: st.error(f"Erreur PyVips: {e}. La bibliothèque système libvips est-elle bien installée ?"); return None, None
    
def get_crop_from_large_file(image_path, box_coords):
    """Extrait la zone de la CNI depuis le fichier volumineux sur disque."""
    x1, y1, x2, y2 = map(int, box_coords)
    crop_vips = pyvips.Image.new_from_file(image_path).crop(x1, y1, x2 - x1, y2 - y1)
    crop_np = np.ndarray(buffer=crop_vips.write_to_memory(), dtype=np.uint8, shape=[crop_vips.height, crop_vips.width, crop_vips.bands])
    return Image.fromarray(crop_np)

# --- PIPELINE D'IA MULTIMODAL (Fusion OCR + Analyse) ---
@st.cache_data(show_spinner=False)
def get_kyc_analysis_from_image(_llm_client, image_bytes):
    """Fait un UNIQUE appel à l'IA multimodale pour l'OCR, l'authentification ET l'extraction."""
    print("Appel à l'API Mistral Multimodale pour analyse complète...")
    kyc_prompt = """
    Tu es un auditeur expert en conformité et en analyse de documents forensiques pour le secteur bancaire de la zone CEMAC. 
    Ta mission est d'analyser l'image d'une Carte Nationale d'Identité (CNI) camerounaise pour un processus KYC critique.
    Ton analyse doit suivre un raisonnement en trois étapes, mais tu ne dois fournir que l'objet JSON final.
    1.  **Phase d'Authentification (Analyse Forensique)** :
        - Vérifie la présence et la clarté des éléments de sécurité textuels : 'REPUBLIQUE DU CAMEROUN', 'NATIONAL IDENTITY CARD'.
        - Analyse la cohérence des polices de caractères. Y a-t-il des fontes mixtes ou d'apparences suspectes ?
        - Évalue l'alignement des champs de texte. Sont-ils droits et professionnels ?
        - Valide la logique des dates (délivrance < expiration).
        - Confirme la présence d'une signature.
        - À partir de ces points, établis un 'score_de_confiance' (0-100) et une 'recommandation' claire : "Approbation Suggérée", "Examen Manuel Approfondi Requis", ou "Rejet Fortement Suggéré".
    2.  **Phase d'Extraction (Lecture de Données)** : Lis avec une précision absolue tous les champs de la carte.
    3.  **Phase de Rapport (Formatage JSON)** : Structure TOUTE ton analyse dans un unique objet JSON. N'ajoute aucun texte avant ou après.
    Voici la structure JSON exacte que tu dois produire :
    {
      "rapport_authentification": {
        "score_de_confiance": "<int>", "recommandation": "<string>",
        "points_de_verification": [
          {"critere": "Présence En-tête 'REPUBLIQUE DU CAMEROUN'", "statut": "OK" | "Anomalie" | "Non Visible", "observation": "<string>"},
          {"critere": "Cohérence des Polices de Caractères", "statut": "OK" | "Anomalie", "observation": "<string>"},
          {"critere": "Alignement des Champs", "statut": "OK" | "Anomalie", "observation": "<string>"},
          {"critere": "Logique des Dates (Délivrance/Expiration)", "statut": "OK" | "Anomalie" | "N/A", "observation": "<string>"},
          {"critere": "Présence de la Signature", "statut": "OK" | "Anomalie" | "Non Visible", "observation": "<string>"}
        ]
      },
      "fiche_identite": {
        "nom": "<string>", "prenoms": "<string>", "date_naissance": "<string>", "lieu_naissance": "<string>", "sexe": "<string>", 
        "profession": "<string>", "pere": "<string>", "mere": "<string>", "adresse": "<string>", "date_delivrance": "<string>", 
        "date_expiration": "<string>", "identifiant_unique_cni": "<string>", "poste_identification": "<string>"
      }
    }
    Si une information est illisible ou absente, utilise la valeur "Non trouvé".
    """
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    try:
        messages = [{"role": "user", "content": [{"type": "text", "text": kyc_prompt}, {"type": "image_url", "image_url": f"data:image/jpeg;base64,{base64_image}"}]}]
        chat_response = _llm_client.chat.complete(model="mistral-large-latest", messages=messages, response_format={"type": "json_object"})
        return json.loads(chat_response.choices[0].message.content)
    except Exception as e: st.error(f"Erreur lors de l'analyse par l'IA : {e}"); return None

# --- COMPOSANTS D'INTERFACE PROFESSIONNELS ---
def display_verification_summary(auth_report):
    """Affiche le verdict et le score de confiance."""
    st.subheader("Synthèse de la Vérification", anchor=False)
    score = auth_report.get('score_de_confiance', 0)
    reco = auth_report.get('recommandation', 'Erreur')
    if reco == "Approbation Suggérée": st.success(f"**Recommandation :** {reco}", icon="✅")
    elif reco == "Examen Manuel Approfondi Requis": st.warning(f"**Recommandation :** {reco}", icon="⚠️")
    else: st.error(f"**Recommandation :** {reco}", icon="🚨")
    st.progress(score, text=f"**Score de Confiance du Document : {score}%**")

def display_authentication_details(auth_report):
    """Affiche la checklist de l'analyse forensique."""
    st.subheader("Analyse d'Authenticité", anchor=False)
    with st.expander("Afficher les points de contrôle forensiques", expanded=False):
        for point in auth_report.get('points_de_verification', []):
            col1, col2, col3 = st.columns([2,1,3])
            with col1: st.markdown(f"**{point['critere']}**")
            with col2:
                statut = point['statut']
                if statut == "OK": st.markdown(f"✅ **{statut}**")
                else: st.markdown(f"⚠️ **{statut}**")
            with col3: st.caption(point['observation'])

def display_identity_card(data):
    """Affiche la fiche d'identité de manière claire et professionnelle."""
    st.subheader("Fiche d'Identité", anchor=False)
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            st.text_input("Nom", value=data.get('nom', 'N/A'), disabled=True, key="nom")
            st.text_input("Prénoms", value=data.get('prenoms', 'N/A'), disabled=True, key="prenoms")
            st.text_input("Sexe", value=data.get('sexe', 'N/A'), disabled=True, key="sexe")
        with col2:
            st.text_input("Date de Naissance", value=data.get('date_naissance', 'N/A'), disabled=True, key="date_naissance")
            st.text_input("Lieu de Naissance", value=data.get('lieu_naissance', 'N/A'), disabled=True, key="lieu_naissance")
        st.text_input("Profession", value=data.get('profession', 'N/A'), disabled=True, key="profession")
        st.divider()
        col3, col4 = st.columns(2);
        with col3: st.text_input("Père", value=data.get('pere', 'N/A'), disabled=True, key="pere")
        with col4: st.text_input("Mère", value=data.get('mere', 'N/A'), disabled=True, key="mere")
        st.divider()
        st.text_input("Poste d'identification", value=data.get('poste_identification', 'N/A'), disabled=True, key="poste_identification")
        col5, col6, col7 = st.columns(3)
        with col5: st.text_input("Délivrance", value=data.get('date_delivrance', 'N/A'), disabled=True, key="date_delivrance")
        with col6: st.text_input("Expiration", value=data.get('date_expiration', 'N/A'), disabled=True, key="date_expiration")
        with col7: st.text_input("Identifiant Unique", value=data.get('identifiant_unique_cni', 'N/A'), disabled=True, key="identifiant_unique_cni")

# --- APPLICATION PRINCIPALE ORCHESTRATRICE ---
def main():
    st.set_page_config(page_title="Auto KYC | Secteur Bancaire", layout="wide", initial_sidebar_state="collapsed")
    st.title("🆔 Assistant de Vérification KYC")
    st.markdown("Outil d'aide à la décision pour l'analyse et la vérification des Cartes Nationales d'Identité.")
    
    detection_model, llm_client = load_detection_model(), load_llm_client()
    if not (detection_model and llm_client): st.stop()
        
    st.divider()
    uploaded_file = st.file_uploader("Chargez une image de haute qualité d'une CNI (recto ou verso)", type=["jpg", "jpeg", "png", "tif", "tiff"])

    if uploaded_file is not None:
        if st.button("Lancer la Vérification ✨", type="primary", use_container_width=True):
            st.session_state.clear()
            with st.spinner("Traitement sécurisé du document en cours..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp:
                    tmp.write(uploaded_file.getvalue()); tmp.flush(); tmp_path = tmp.name
                
                with st.status("Étape 1/3 : Localisation du document...", expanded=True) as status:
                    global_box, thumbnail_img = process_large_image_in_tiles(detection_model, tmp_path)
                    st.session_state.processed_img = thumbnail_img
                    if global_box is None:
                        status.update(label="Localisation échouée.", state="error", expanded=False)
                        st.error("Aucun document d'identité n'a pu être localisé sur l'image."); st.stop()
                    status.update(label="Document localisé.", state="complete", expanded=False)

                with st.status("Étape 2/3 : Analyse par l'IA...", expanded=True) as status:
                    cni_crop_pil = get_crop_from_large_file(tmp_path, global_box)
                    with io.BytesIO() as buf: cni_crop_pil.save(buf, format='JPEG', quality=95); image_bytes = buf.getvalue()
                    final_report = get_kyc_analysis_from_image(llm_client, image_bytes)
                    st.session_state.final_report = final_report
                    if final_report is None:
                        status.update(label="Analyse IA échouée.", state="error", expanded=False)
                        st.error("L'analyse par l'IA a échoué."); st.stop()
                    status.update(label="Analyse IA terminée.", state="complete", expanded=False)
                os.remove(tmp_path) # Nettoyer le fichier temporaire

    if 'final_report' in st.session_state and st.session_state.final_report:
        st.divider(); st.header("Rapport de Vérification KYC", anchor=False)
        report = st.session_state.final_report
        col_summary, col_preview = st.columns([2, 1])
        with col_summary:
            display_verification_summary(report.get("rapport_authentification", {}))
            display_authentication_details(report.get("rapport_authentification", {}))
        with col_preview:
            pil_to_display = Image.fromarray(cv2.cvtColor(st.session_state.processed_img, cv2.COLOR_BGR2RGB))
            st.image(pil_to_display, caption="Aperçu du document analysé", use_container_width=True)
        st.divider(); display_identity_card(report.get("fiche_identite", {}))

if __name__ == "__main__":
    main()