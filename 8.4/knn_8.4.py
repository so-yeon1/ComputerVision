import cv2
import numpy as np
import os
from memory_profiler import profile

## knn 모델 저장 및 크기 반환
def model_size(model, filename):
    model.save(filename)
    return os.path.getsize(filename)

## 학습 전 knn 크기
def initial_knn(knn):
    initial_size = model_size(knn, 'knn_initial.xml')
    print("[학습 전]")
    print(f"knn 크기: {initial_size} bytes")
    print()
    return knn

## 학습 후 knn 크기
def trained_knn(knn):
    trained_size = model_size(knn, 'knn_trained.xml')
    print("[학습 후]")
    print(f"knn 크기: {trained_size} bytes")

@profile    # 메모리 사용량 추적
def main():
    L=20    # 학습데이터 수
    data = np.random.randint(0, 100, (L, 2)).astype(np.float32)
    labels = np.random.randint(0, 2, (L, 1)).astype(np.int32)

    # KNN 모델 생성 및 크기 측정
    knn = cv2.ml.KNearest_create()
    knn = initial_knn(knn)

    # KNN 모델 학습 및 크기 측정
    knn.train(data, cv2.ml.ROW_SAMPLE, labels)
    trained_knn(knn)

if __name__ == "__main__":
	main()

