import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

# --- Transformations pour l'entraînement (avec augmentation) ---
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),           # taille standard pour les modèles pré-entraînés
    transforms.RandomHorizontalFlip(p=0.5),  # flip horizontal aléatoire
    transforms.RandomRotation(15),           # rotation légère aléatoire
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # variation de luminosité/contraste
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # normalisation standard ImageNet
                          std=[0.229, 0.224, 0.225])
])

# --- Transformations pour la validation (pas d'augmentation, juste resize + normalize) ---
val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                          std=[0.229, 0.224, 0.225])
])

# --- Charger le dataset complet ---
full_dataset = datasets.ImageFolder(root="data", transform=train_transform)

print("Classes détectées:", full_dataset.classes)
print("Nombre total d'images:", len(full_dataset))

# --- Split train / validation (80% / 20%) ---
val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

print(f"Train: {train_size} images | Validation: {val_size} images")

# --- Calcul des poids de classes (pour compenser le déséquilibre) ---
class_counts = [0] * len(full_dataset.classes)
for _, label in full_dataset.samples:
    class_counts[label] += 1

class_weights = [1.0 / c for c in class_counts]
print("Poids par classe:", dict(zip(full_dataset.classes, [round(w, 4) for w in class_weights])))

# --- DataLoaders ---
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# --- Test rapide : afficher un batch ---
images, labels = next(iter(train_loader))
print("\nUn batch d'entraînement:")
print("Shape des images:", images.shape)  # [batch_size, 3, 224, 224]
print("Labels:", labels)