import sys
import glob
import time
sys.path.insert(0, "src")

import torch
from PIL import Image
from torchvision import transforms
from model import creer_modele

classes = ['5laal', '8ta', '9ola', 'bal8a', 'nawar', 'r7a', 'rob', 'thoub']

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

model = creer_modele(nb_classes=len(classes))
model.load_state_dict(torch.load("models/resnet18_pieces_archeo.pth", map_location="cpu"))
model.eval()

image_path = glob.glob("data/rob/*")[0]
print("Image utilisee :", image_path)

N = 50
with torch.no_grad():
    x = transform(Image.open(image_path).convert("RGB")).unsqueeze(0)
    for _ in range(5):          # echauffement
        model(x)

    debut = time.perf_counter()
    for _ in range(N):
        model(x)
    t_modele = (time.perf_counter() - debut) / N * 1000

    debut = time.perf_counter()
    for _ in range(N):
        x = transform(Image.open(image_path).convert("RGB")).unsqueeze(0)
        model(x)
    t_complet = (time.perf_counter() - debut) / N * 1000

print(f"Passage dans le modele seul : {t_modele:.1f} ms")
print(f"Lecture + pretraitement + modele : {t_complet:.1f} ms")