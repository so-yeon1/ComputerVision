import cv2
import numpy as np
from collections import Counter

L = 4000        # 학습데이터 개수
maxL = 200      # 데이터 범위

if L >= maxL**2:
    print(f'생성할 수 있는 학습데이터의 수는 {maxL**2}개 미만입니다.')
    exit(0)

# 중복 없는 학습 데이터 생성
data = np.random.randint(0, maxL, (L, 2)).astype(np.float32)
data = np.unique(data, axis=0)

while len(data) < L:       # 지정한 학습 데이터 수만큼 중복 없는 데이터 생성
    newData = np.random.randint(0,maxL,(L-len(data),2)).astype(np.float32)
    data = np.append(data, newData, axis=0)
    data = np.unique(data, axis=0)

# 레이블 생성
labels = np.random.randint(0, 2, (L, 1)).astype(np.float32)

# knn 모델 생성
knn = cv2.ml.KNearest_create()

# knn 학습
knn.train(data, cv2.ml.ROW_SAMPLE, labels)

K = [1]

def get_accuracy(predictions, labels):
    accuracy = (np.squeeze(predictions) == np.squeeze(labels)).mean()   # predictions와 labels shape 맞춰주기
    return accuracy * 100   # labels가 10개라서 맞힌 갯수를 평균(mean)을 구해서 100을 곱하면 정확도가 나온다.

for k in K:
    # knn 분류 test
    ret_val = knn.findNearest(data, k)
    ret, results, neighbours, dist = ret_val

    cmp = labels == results
    cmp_f = cmp.flatten()
    dict = Counter(cmp_f)
    print(f'test=train: L={L}, k={k}: Accuracy={dict[True] * 100 / len(cmp):#.2f}%')

    # 다른 예제에서 사용하였던 함수로 accuracy를 구해본다.
    acc = get_accuracy(results, labels)
    print(f"test=train: L={L}, k={k}: Accuracy2={acc:#6.2f}%")

exit(0)
