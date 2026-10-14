import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

# --- Graine aléatoire fixée (reproductibilité) ---
SEED = 42
torch.manual_seed(SEED)

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

# --- Deux datasets sur les mêmes images, avec des transformations différentes ---
train_full = datasets.ImageFolder(root="data", transform=train_transform)
val_full = datasets.ImageFolder(root="data", transform=val_transform)
full_dataset = train_full  # nom conservé pour les autres scripts

# --- Split train / validation (80% / 20%), stratifié et reproductible ---
targets = train_full.targets
train_idx, val_idx = train_test_split(
    list(range(len(targets))),
    test_size=0.2, stratify=targets, random_state=SEED)

train_dataset = Subset(train_full, train_idx)  # avec augmentation
val_dataset = Subset(val_full, val_idx)        # sans augmentation

train_size = len(train_idx)
val_size = len(val_idx)

# --- Calcul des poids de classes : N / (K * n_c), sur l'ensemble d'entraînement ---
nb_classes = len(train_full.classes)
class_counts = [0] * nb_classes
for i in train_idx:
    class_counts[targets[i]] += 1

class_weights = torch.tensor(
    [train_size / (nb_classes * c) for c in class_counts],
    dtype=torch.float)

# --- DataLoaders ---
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# --- Test rapide (uniquement si on lance ce fichier directement) ---
if __name__ == "__main__":
    print("Classes détectées:", train_full.classes)
    print("Nombre total d'images:", len(train_full))
    print(f"Train: {train_size} images | Validation: {val_size} images")
    print("Poids par classe:", dict(zip(train_full.classes,
                                        [round(w, 4) for w in class_weights.tolist()])))

    images, labels = next(iter(train_loader))
    print("\nUn batch d'entraînement:")
    print("Shape des images:", images.shape)  # [batch_size, 3, 224, 224]
    print("Labels:", labels)