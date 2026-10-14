import os
import torch
import torch.nn as nn
import torch.optim as optim

from model import creer_modele
from load_dataset import full_dataset, train_loader, val_loader, class_weights, train_size, val_size

classes = full_dataset.classes
print(f"Train: {train_size} images | Validation: {val_size} images")

# --- Modèle, perte, optimiseur ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = creer_modele(nb_classes=len(classes)).to(device)
class_weights = class_weights.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.fc.parameters(), lr=0.001)  # on entraîne seulement la dernière couche

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

    train_loss = running_loss / len(train_loader)  # perte moyenne par lot
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

    print(f"Epoch {epoch+1}/{NB_EPOCHS} - Loss: {train_loss:.4f} - Train Acc: {train_acc:.1f}% - Val Acc: {val_acc:.1f}%")

# --- Sauvegarder le modèle entraîné ---
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/resnet18_pieces_archeo.pth")
print("\nModele sauvegarde dans models/resnet18_pieces_archeo.pth")