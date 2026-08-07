from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image
import io
import sys
import os

# Permet d'importer model.py depuis src/
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from model import creer_modele

app = FastAPI(title="API Identification Pièces Archéologiques")

# Autorise les requêtes depuis n'importe quelle origine (utile pour tester depuis une app mobile/web)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

classes = ['5laal', '8ta', '9ola', 'bal8a', 'nawar', 'r7a', 'rob', 'thoub']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Charger le modèle une seule fois au démarrage de l'API ---
model = creer_modele(nb_classes=len(classes))
model_path = os.path.join(os.path.dirname(__file__), "..", "models", "resnet18_pieces_archeo.pth")
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

@app.get("/")
def accueil():
    return {"message": "API d'identification de pièces archéologiques - Musée de Gabès"}

@app.post("/predict")
async def predire_image(file: UploadFile = File(...)):
    contenu = await file.read()
    image = Image.open(io.BytesIO(contenu)).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)

    return {
        "classe_predite": classes[predicted_idx.item()],
        "confiance": round(confidence.item() * 100, 1)
    }