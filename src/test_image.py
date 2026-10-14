import cv2
import os

image_path = os.path.join("data", "5laal", "IMG_20260805_095155.jpg")

img = cv2.imread(image_path)

if img is None:
    print("Erreur : image introuvable. Vérifie le nom et le chemin.")
else:
    print("Image chargée avec succès !")
    print("Dimensions (hauteur, largeur, canaux):", img.shape)
    
    # Redimensionner pour l'affichage (garde l'original intact)
    img_resized = cv2.resize(img, (800, 600))
    
    cv2.imshow("Piece archeologique", img_resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    