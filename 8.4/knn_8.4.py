'''
4조_박소연, 강희정, 이태훈

* 프로그램 목적:
1) knn 모델은 학습한 데이터를 모두 저장하는지를 알아본다.
2) knn 모델의 크기가 학습 데이터의 양에 따라 달라지는지를 알아본다.

* 작동 방법:
1) 학습데이터의 수를 설정하고, 데이터를 생성한다.
2) knn 모델을 생성한다.
3) 학습 전 knn모델을 파일로 저장하고, 파일 크기를 출력한다.
    -> model.save(filename): 파일저장
       os.path.getsize(filename): 파일 크기 반환
4) knn 학습을 수행한다.
5) 학습 후 knn모델을 파일로 저장하고, 파일 크기를 출력한다.


=> 1. 학습후 knn 모델 파일 내부에 학습한 데이터가 저장되었는지 확인한다.
   2. 학습 데이터의 수를 변경해보며 데이터의 양에 따른 파일 크기 변화를 확인한다.

'''

import cv2
import numpy as np
import os

## knn 모델 저장 및 크기 반환
def model_size(model, filename):
    model.save(filename)    # 모델 저장
    return os.path.getsize(filename)    # 파일 크기 반환

## 학습 전 knn 크기
def initial_knn(knn):
    initial_size = model_size(knn, 'knn_initial.xml')
    print(f"학습 전 knn 크기: {initial_size} bytes")
    return knn

## 학습 후 knn 크기
def trained_knn(knn):
    trained_size = model_size(knn, 'knn_trained.xml')
    print(f"학습 후 knn 크기: {trained_size} bytes")


L=20    # 학습데이터 수
data = np.random.randint(0, 100, (L, 2)).astype(np.float32)
labels = np.random.randint(0, 2, (L, 1)).astype(np.int32)

# KNN 모델 생성 및 크기 측정
knn = cv2.ml.KNearest_create()
knn = initial_knn(knn)

# KNN 모델 학습 및 크기 측정
knn.train(data, cv2.ml.ROW_SAMPLE, labels)
trained_knn(knn)