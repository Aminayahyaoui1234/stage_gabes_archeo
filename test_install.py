import torch
import cv2

print("PyTorch version:", torch.__version__)
print("OpenCV version:", cv2.__version__)
print("CUDA disponible:", torch.cuda.is_available())