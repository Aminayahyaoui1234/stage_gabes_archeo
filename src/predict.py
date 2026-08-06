import torch
from torchvision import transforms
from PIL import Image
import os

from model import creer_modele

# --- Les mêmes classes que pendant l'entraînement (ordre alphabétique) ---
classes = ['5laal', '8ta', '9ola', 'bal8a', 'nawar', 'r7a', 'rob', 'thoub']

# --- Même transformation que la validation (pas d'augmentation) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def charger_modele():
    model = creer_modele(nb_classes=len(classes))
    model.load_state_dict(torch.load("models/resnet18_pieces_archeo.pth", map_location="cpu"))
    model.eval()
    return model

def predire(image_path, model):
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0]
        confidence, predicted_idx = torch.max(probabilities, 0)

    classe_predite = classes[predicted_idx.item()]
    return classe_predite, confidence.item()

if __name__ == "__main__":
    model = charger_modele()

    image_test = os.path.join("data", "8ta", "IMG_20260805_095404.jpg")

    classe, confiance = predire(image_test, model)
    print(f"Image: {image_test}")
    print(f"Classe prédite: {classe}")
    print(f"Confiance: {confiance*100:.1f}%")