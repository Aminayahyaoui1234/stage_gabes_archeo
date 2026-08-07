import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from model import creer_modele

# --- Mêmes transformations que pour la validation (pas d'augmentation) ---
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Recharger le même split qu'à l'entraînement ---
full_dataset = datasets.ImageFolder(root="data", transform=val_transform)
classes = full_dataset.classes

val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size

# IMPORTANT : même seed que train.py pour retrouver le même split
torch.manual_seed(42)
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# --- Charger le modèle entraîné ---
model = creer_modele(nb_classes=len(classes))
model.load_state_dict(torch.load("models/resnet18_pieces_archeo.pth", map_location="cpu"))
model.eval()

print(f"Évaluation sur {len(val_dataset)} images de validation\n")
print("Classes:", classes)
# --- Collecter toutes les prédictions et vraies étiquettes ---
all_preds = []
all_labels = []

with torch.no_grad():
    for images, labels in val_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.tolist())
        all_labels.extend(labels.tolist())

# --- Construire la matrice de confusion manuellement ---
nb_classes = len(classes)
matrice = [[0] * nb_classes for _ in range(nb_classes)]

for vrai, predit in zip(all_labels, all_preds):
    matrice[vrai][predit] += 1

# --- Affichage lisible ---
print("\nMatrice de confusion (lignes = vraie classe, colonnes = classe prédite)\n")

en_tete = "          " + "".join(f"{c[:6]:>8}" for c in classes)
print(en_tete)

for i, ligne in enumerate(matrice):
    texte_ligne = f"{classes[i][:8]:<10}" + "".join(f"{val:>8}" for val in ligne)
    print(texte_ligne)

# --- Précision globale ---
correct = sum(matrice[i][i] for i in range(nb_classes))
total = sum(sum(ligne) for ligne in matrice)
print(f"\nPrécision globale sur validation: {100 * correct / total:.1f}% ({correct}/{total})")