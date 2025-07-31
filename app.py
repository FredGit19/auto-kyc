# ==============================================================================
# APPLICATION STREAMLIT - VERSION FINALE ET ROBUSTE
# ==============================================================================
# Corrections et améliorations clés :
# - COHÉRENCE TOTALE : Utilise Albumentations pour le pré-traitement, comme dans 
#   le script d'entraînement, afin d'éliminer toute désynchronisation.
# - ROBUSTESSE : Gestion améliorée des erreurs de chargement et de détection.
# - CLARTÉ : Code et commentaires améliorés pour une meilleure compréhension.
# ==============================================================================

import streamlit as st
import torch
import cv2
import numpy as np
import easyocr
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# --- CONFIGURATION ET CHARGEMENT DES MODÈLES (Sécurisé par cache) ---

MODEL_PATH = "models/frcnn_cni_best.pth" # Assurez-vous que le chemin est correct
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFIDENCE_THRESHOLD = 0.7 # Seuil de confiance

def get_model_architecture(num_classes):
    """
    DÉFINITION IDENTIQUE À L'ENTRAÎNEMENT :
    Construit l'architecture exacte du modèle pour pouvoir charger les poids.
    """
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

@st.cache_resource
def load_detection_model():
    """Charge le modèle de détection Faster R-CNN entraîné."""
    try:
        # 1. Créer l'architecture vide
        model = get_model_architecture(num_classes=2) # 1 classe (CNI) + 1 fond
        
        # 2. Charger le checkpoint (poids, etc.)
        checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
        
        # 3. Charger les poids dans l'architecture
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model.to(DEVICE)
        model.eval() # Mettre le modèle en mode évaluation (très important !)
        print("Modèle de détection chargé avec succès sur le device:", DEVICE)
        return model
    except FileNotFoundError:
        st.error(f"ERREUR CRITIQUE : Le fichier du modèle est introuvable au chemin '{MODEL_PATH}'.")
        st.error("Vérifiez que le modèle 'frcnn_cni_best.pth' se trouve bien dans le dossier 'models/'.")
        return None
    except Exception as e:
        st.error(f"Une erreur est survenue lors du chargement du modèle de détection : {e}")
        return None

@st.cache_resource
def load_ocr_model():
    """Charge le modèle EasyOCR pour la reconnaissance de texte."""
    try:
        reader = easyocr.Reader(['fr', 'en'], gpu=torch.cuda.is_available())
        print("Modèle OCR chargé avec succès.")
        return reader
    except Exception as e:
        st.error(f"Une erreur est survenue lors du chargement du modèle OCR : {e}")
        return None

# --- FONCTIONS DE TRAITEMENT AVEC LA CORRECTION DÉFINITIVE ---

def get_inference_transforms():
    """
    LA CORRECTION CLÉ : Utilise le même pipeline de validation que l'entraînement.
    """
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # Normalisation ImageNet
        ToTensorV2(), # Convertit en Tensor PyTorch
    ])

def detect_cni(model, pil_image):
    """
    Effectue la détection sur une image en utilisant le pipeline de transformation correct.
    """
    if model is None: return None, None
        
    # 1. Convertir l'image PIL en tableau NumPy, format attendu par Albumentations
    image_np = np.array(pil_image.convert('RGB'))
    
    # 2. Appliquer les transformations
    transforms = get_inference_transforms()
    transformed = transforms(image=image_np)
    image_tensor = transformed['image'].to(DEVICE)
    
    # 3. Le modèle attend un "batch" d'images, même s'il n'y en a qu'une
    image_tensor = image_tensor.unsqueeze(0) 
    
    with torch.no_grad():
        prediction = model(image_tensor)
        
    # 4. Extraire et filtrer les résultats
    boxes = prediction[0]['boxes'].cpu().numpy()
    scores = prediction[0]['scores'].cpu().numpy()
    
    best_box = None
    max_score = 0
    # On prend la boîte avec le plus haut score au-dessus du seuil
    for box, score in zip(boxes, scores):
        if score > CONFIDENCE_THRESHOLD and score > max_score:
            max_score = score
            best_box = box

    # Dessiner le résultat sur l'image originale (au format OpenCV)
    image_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    if best_box is not None:
        x1, y1, x2, y2 = map(int, best_box)
        cv2.rectangle(image_cv, (x1, y1), (x2, y2), (36, 255, 12), 2) # Couleur verte vive
        label = f"CNI: {max_score:.2f}"
        cv2.putText(image_cv, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)
    
    return image_cv, best_box

def parse_ocr_results(ocr_results):
    """
    Tente de parser les résultats bruts d'EasyOCR. C'est une heuristique.
    """
    # ... (le code de parsing reste le même, il est à adapter selon les résultats)
    extracted_data = {"NOM": "Non trouvé", "PRENOMS": "Non trouvé", "DATE_NAISSANCE": "Non trouvé"}
    full_text = " ".join([res[1] for res in ocr_results])
    
    for i, res in enumerate(ocr_results):
        text = res[1].upper()
        if "NOM" in text and i + 1 < len(ocr_results):
            extracted_data["NOM"] = ocr_results[i+1][1]
        if "PRENOMS" in text and i + 1 < len(ocr_results):
            extracted_data["PRENOMS"] = ocr_results[i+1][1]
        if "NEE LE" in text or "NE LE" in text and i + 1 < len(ocr_results):
            extracted_data["DATE_NAISSANCE"] = ocr_results[i+1][1]
            
    return extracted_data, full_text

# --- INTERFACE UTILISATEUR STREAMLIT ---

def main():
    st.set_page_config(page_title="Analyseur de CNI", layout="wide", initial_sidebar_state="auto")
    st.title("🤖 Analyseur de Carte Nationale d'Identité")
    st.markdown("---")

    st.sidebar.header("Instructions")
    st.sidebar.info(
        "1. Chargez une image claire et bien cadrée de la CNI.\n\n"
        "2. Cliquez sur le bouton 'Lancer l'analyse'.\n\n"
        "3. L'application va d'abord localiser la carte (boîte verte), puis lire les informations qu'elle contient."
    )

    uploaded_file = st.file_uploader(
        "Déposez une image ici ou cliquez pour parcourir vos fichiers", 
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        col1, col2 = st.columns(2)
        pil_image = Image.open(uploaded_file)
        
        with col1:
            st.subheader("Image Originale")
            st.image(pil_image, use_column_width=True)

        if st.button("🚀 Lancer l'analyse", use_container_width=True, type="primary"):
            detection_model = load_detection_model()
            ocr_model = load_ocr_model()

            if detection_model and ocr_model:
                with col2:
                    st.subheader("Résultats de l'Analyse")
                    with st.spinner("Étape 1/2 : Détection de la carte en cours..."):
                        detected_image, box = detect_cni(detection_model, pil_image)
                    
                    if box is None:
                        st.warning("Aucune CNI n'a pu être détectée avec certitude. Essayez une image plus nette ou mieux cadrée.")
                        st.image(detected_image, channels="BGR", use_column_width=True, caption="Tentative de détection.")
                    else:
                        st.success("✅ CNI localisée !")
                        st.image(detected_image, channels="BGR", use_column_width=True, caption="CNI détectée sur l'image.")
                        
                        with st.spinner("Étape 2/2 : Lecture des informations (OCR)..."):
                            x1, y1, x2, y2 = map(int, box)
                            cni_crop = pil_image.crop((x1, y1, x2, y2))
                            ocr_results = ocr_model.readtext(np.array(cni_crop))
                            structured_data, raw_text = parse_ocr_results(ocr_results)
                        
                        st.success("✅ Lecture terminée !")
                        st.write("#### Informations Extraites")
                        st.json(structured_data)
                        
                        with st.expander("Afficher le texte brut extrait"):
                            st.text_area("", raw_text, height=150)

if __name__ == "__main__":
    main()