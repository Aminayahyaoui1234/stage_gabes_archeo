import os
import torch
from sklearn.metrics import classification_report, confusion_matrix

from model import creer_modele
from load_dataset import full_dataset, val_loader, val_size

classes = full_dataset.classes

# --- Charger le modèle entraîné ---
model = creer_modele(nb_classes=len(classes))
model.load_state_dict(torch.load("models/resnet18_pieces_archeo.pth", map_location="cpu"))
model.eval()

print(f"Evaluation sur {val_size} images de validation\n")
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

# --- Matrice de confusion ---
indices = list(range(len(classes)))
matrice = confusion_matrix(all_labels, all_preds, labels=indices)

print("\nMatrice de confusion (lignes = vraie classe, colonnes = classe predite)\n")
print("          " + "".join(f"{c[:6]:>8}" for c in classes))
for i, ligne in enumerate(matrice):
    print(f"{classes[i][:8]:<10}" + "".join(f"{int(v):>8}" for v in ligne))

# --- Précision globale ---
correct = int(sum(matrice[i][i] for i in indices))
total = int(matrice.sum())
print(f"\nPrecision globale sur validation: {100 * correct / total:.1f}% ({correct}/{total})")

# --- Précision, rappel, F1 par classe ---
print("\nMetriques par classe\n")
print(classification_report(all_labels, all_preds, labels=indices,
                            target_names=classes, digits=3, zero_division=0))

# --- Sauvegarde de la matrice de confusion en image ---
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    os.makedirs("image/ia_muusee", exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(matrice, display_labels=classes).plot(
        ax=ax, cmap="Blues", colorbar=False)
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Vraie classe")
    plt.xticks(rotation=45)
    plt.tight_layout()
    fig.savefig("image/ia_muusee/confusion.png", dpi=200)
    print("Image enregistree : image/ia_muusee/confusion.png")
except ImportError:
    print("matplotlib n'est pas installe : lancer 'pip install matplotlib' puis relancer.")