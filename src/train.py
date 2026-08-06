import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split

from model import creer_modele

# --- Transformations (mêmes que load_dataset.py) ---
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Dataset et split ---
full_dataset = datasets.ImageFolder(root="data", transform=train_transform)
classes = full_dataset.classes

val_size = int(0.2 * len(full_dataset))
train_size = len(full_dataset) - val_size
train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)

# --- Poids de classes pour compenser le déséquilibre ---
class_counts = [0] * len(classes)
for _, label in full_dataset.samples:
    class_counts[label] += 1
class_weights = torch.tensor([1.0 / c for c in class_counts], dtype=torch.float)

# --- Modèle, perte, optimiseur ---
model = creer_modele(nb_classes=len(classes))
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # on entraîne seulement la dernière couche

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
class_weights = class_weights.to(device)

# --- Boucle d'entraînement ---
NB_EPOCHS = 15

for epoch in range(NB_EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = 100 * correct / total

    # --- Validation ---
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_acc = 100 * val_correct / val_total

    print(f"Epoch {epoch+1}/{NB_EPOCHS} - Loss: {running_loss:.4f} - Train Acc: {train_acc:.1f}% - Val Acc: {val_acc:.1f}%")

# --- Sauvegarder le modèle entraîné ---
torch.save(model.state_dict(), "models/resnet18_pieces_archeo.pth")
print("\nModèle sauvegardé dans models/resnet18_pieces_archeo.pth")