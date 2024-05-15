import cv2
import numpy as np
import os
import sys
from pympler import asizeof
from memory_profiler import profile
import psutil

def model_size(model, filename):
    model.save(filename)
    pid = os.getpid()
    process = psutil.Process(pid)
    memory = process.memory_info().rss
    memory_size = memory
    # print(f"사용 중인 메모리: {memory / 1024 ** 2}MiB")

    return os.path.getsize(filename), asizeof.asizeof(model) # sys.getsizeof(model) #memory_size

def initial_knn(knn):
    initial_size, initial_mem = model_size(knn, 'knn_initial.xml')
    print("[학습 전]")
    print(f"knn 크기: {initial_size} bytes")
    print(f"knn 메모리: {initial_mem} bytes")
    print()
    return knn

def trained_knn(knn):
    trained_size, trained_mem = model_size(knn, 'knn_trained.xml')
    print("[학습 후]")
    print(f"knn 크기: {trained_size} bytes")
    print(f"knn 메모리: {trained_mem} bytes")

@profile
def main():
    L=20
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
