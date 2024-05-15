import cv2
import numpy as np
import os
import sys
from pympler import asizeof

def save_model(model, filename):
    model.save(filename)
    return os.path.getsize(filename), asizeof.asizeof(filename)

def create_knn_model():
    knn = cv2.ml.KNearest_create()
    initial_size, initial_mem = save_model(knn, 'knn_initial.xml')
    print(f"Initial KNN model size: {initial_size} bytes")
    print(f"Initial KNN model memory: {initial_mem} bytes")
    print()
    return knn

def train_knn_model(knn, features, labels):
    knn.train(features, cv2.ml.ROW_SAMPLE, labels)
    trained_size, trained_mem = save_model(knn, 'knn_trained.xml')
    print(f"Trained KNN model size: {trained_size} bytes")
    print(f"Trained KNN model memory: {trained_mem} bytes")

def main():
    L=20000
    data = np.random.randint(0, 100, (L, 2)).astype(np.float32)
    labels = np.random.randint(0, 2, (L, 1)).astype(np.int32)

    # KNN 모델 생성 및 크기 측정
    knn = create_knn_model()

    # KNN 모델 학습 및 크기 측정
    train_knn_model(knn, data, labels)

if __name__ == "__main__":
    main()