import os
import cv2
import numpy as np

DATASET_DIR = "dataset"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_user(user_folder):
    path = os.path.join(DATASET_DIR, user_folder)
    images, labels = [], []

    for file in os.listdir(path):
        img_path = os.path.join(path, file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        img = cv2.resize(img, (200, 200))
        images.append(img)
        labels.append(0)  

    if images:
        model = cv2.face.LBPHFaceRecognizer_create()
        model.train(images, np.array(labels))
        model.save(os.path.join(MODEL_DIR, f"{user_folder}.yml"))

for user in os.listdir(DATASET_DIR):
    if os.path.isdir(os.path.join(DATASET_DIR, user)):
        train_user(user)
