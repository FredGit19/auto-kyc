# convert_model.py

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# --- Configuration ---
OLD_MODEL_PATH = "models/frcnn_cni_best.pth"
NEW_MODEL_PATH = "models/frcnn_cni_best_safe.pth" # Le nom du nouveau fichier sécurisé
NUM_CLASSES = 2 # 1 classe (CNI) + 1 fond

print("--- Démarrage de la conversion du modèle ---")

# --- Étape 1 : Construire l'architecture du modèle ---
print(f"1. Construction de l'architecture du modèle pour {NUM_CLASSES} classes...")
model = fasterrcnn_resnet50_fpn(weights=None)
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES)
print("   Architecture construite avec succès.")

# --- Étape 2 : Charger l'ancien checkpoint ---
# On utilise l'ancienne méthode non sécurisée ici, car c'est notre propre fichier.
print(f"2. Chargement de l'ancien checkpoint depuis '{OLD_MODEL_PATH}'...")
try:
    # On utilise torch.load SANS le paramètre weights_only pour être compatible avec l'ancien format
    checkpoint = torch.load(OLD_MODEL_PATH, map_location=torch.device('cpu'))
    # On charge les poids dans le modèle
    model.load_state_dict(checkpoint['model_state_dict'])
    print("   Ancien checkpoint chargé avec succès.")
except Exception as e:
    print(f"\nERREUR : Impossible de charger l'ancien modèle. Assurez-vous que le fichier '{OLD_MODEL_PATH}' est correct.")
    print(f"Détails de l'erreur : {e}")
    exit() # On arrête le script si le chargement échoue

# --- Étape 3 : Sauvegarder les poids dans le nouveau format sécurisé ---
# torch.save(model.state_dict(), ...) ne sauvegarde QUE les poids, ce qui est le format moderne.
print(f"3. Sauvegarde des poids au format sécurisé dans '{NEW_MODEL_PATH}'...")
torch.save(model.state_dict(), NEW_MODEL_PATH)

print("\n--- ✅ Conversion terminée avec succès ! ---")
print(f"Votre nouveau modèle sécurisé est prêt à l'emploi : '{NEW_MODEL_PATH}'")