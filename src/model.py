import torch
import torch.nn as nn
from torchvision import models

def creer_modele(nb_classes=8):
    # Charger ResNet18 pré-entraîné sur ImageNet
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Geler les couches pré-entraînées (on ne les réentraîne pas, on garde ce qu'elles savent déjà)
    for param in model.parameters():
        param.requires_grad = False
    
    # Remplacer la dernière couche (classification) par une adaptée à nos 8 classes
    nb_features = model.fc.in_features
    model.fc = nn.Linear(nb_features, nb_classes)
    # Cette dernière couche, elle, sera bien entraînée
    
    return model

if __name__ == "__main__":
    model = creer_modele(nb_classes=8)
    print(model.fc)
    
    # Test rapide avec un batch factice
    dummy_input = torch.randn(8, 3, 224, 224)
    output = model(dummy_input)
    print("Shape de sortie:", output.shape)  # attendu: [8, 8] -> 8 images, 8 scores de classe chacune