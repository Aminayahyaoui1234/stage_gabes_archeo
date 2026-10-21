import os

data_dir = "data"
print("Aperçu du dataset :\n")

total = 0
for categorie in sorted(os.listdir(data_dir)):
    chemin_categorie = os.path.join(data_dir, categorie)
    if os.path.isdir(chemin_categorie):
        images = [f for f in os.listdir(chemin_categorie)
                  if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        nb = len(images)
        total += nb
        print(f"{categorie}: {nb} images")

nb_categories = len([d for d in os.listdir(data_dir)
                     if os.path.isdir(os.path.join(data_dir, d))])
print(f"\nTotal: {total} images sur {nb_categories} catégories")