# opencv 이용
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ====================================== 단계 1,2 (line 8~65)======================================= #

nData = 100  # 각 클래스의 데이터 수
Range = 100  # 데이터의 전체 범위
nClass = 4  # 클래스 개수 (단계 2)
overlap = 20    # 클래스별 데이터 겹치는 범위

## 데이터 생성
def generateData(L,R,nClass,overlap):
    data = []
    labels = []
    for i in range(nClass):
        start = i * (R // nClass) - overlap      # 겹치는 범위: overlap
        end = start + (R // nClass) + overlap
        class_data = np.random.randint(max(start, 0), min(end, R), (L, 2)).astype(np.float32)   # 클래스 별 데이터 생성
        data.append(class_data)
        class_labels = np.full((L, 1), i, dtype=np.int32)   # label 생성
        labels.append(class_labels)
    return np.vstack(data), np.vstack(labels)   # 클래스별로 나뉜 배열을 하나의 배열로 병합해 리턴

data, labels = generateData(nData, Range, nClass, overlap)

colors = [(255, 0, 0), (255, 0, 255), (0, 0, 255), (217, 65, 128)]      # 데이터(circle) 색상
back_colors = [(229,209,92), (245,178,255), (158,193,255), (255,178,209)]   # 배경 색상

gammas = [0.01, 0.1, 1, 10]
fig, axes = plt.subplots(1, len(gammas), figsize=(15, 3))

## gamma에 따른 SVM분류(단계1)
for gamma, ax in zip(gammas, axes):
    # SVM 설정
    svm_model = cv2.ml.SVM_create()
    svm_model.setKernel(cv2.ml.SVM_RBF)
    svm_model.setType(cv2.ml.SVM_C_SVC)
    svm_model.setGamma(gamma)
    svm_model.setC(10)  # C 값 설정
    svm_model.train(data, cv2.ml.ROW_SAMPLE, labels)

    # 전체 좌표 predict
    a = np.linspace(0, 99, 100)
    b = np.linspace(0, 99, 100)
    x, y = np.meshgrid(a,b)     # 2차원 그리드 생성
    cooldinate = np.c_[x.ravel(), y.ravel()].astype(np.float32)     # x좌표,y좌표를 가로방향으로 병합

    _, response = svm_model.predict(cooldinate)    # 예측
    response = response.reshape(x.shape)

    # 예측 결과 화면에 표시
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    for label in range(nClass):
        img[response == label] = back_colors[label]

    # 학습 데이터 표시
    for loc, label in zip(data, labels):
        cv2.circle(img, (int(loc[0]),int(loc[1])), 1, colors[label[0]], -1)

    RGBimg = img[:, :, ::-1]
    ax.imshow(RGBimg)
    ax.set_title(f'gamma:{gamma}')
    ax.axis('off')



# ================================ 단계 3,4,5 (line 72~130) ===================================== #


nData = 100  # 각 클래스의 데이터 수
Range = 640  # 데이터의 전체 범위
nClass = 4  # 클래스 개수
overlap = 100    # 겹치는 범위

# 데이터 생성
data, labels = generateData(nData, Range, nClass, overlap)

# SVM 모델 생성
svm_model = cv2.ml.SVM_create()
svm_model.setType(cv2.ml.SVM_C_SVC)

kernel_options = [cv2.ml.SVM_RBF, cv2.ml.SVM_LINEAR, cv2.ml.SVM_CHI2, cv2.ml.SVM_INTER, cv2.ml.SVM_SIGMOID, cv2.ml.SVM_POLY]
kernel_name = ['RBF','LINEAR','CHI2','INTER','SIGMOID','POLY']

## SVM 학습 및 예측(트랙바 콜백 함수)
def trainMethod():
    kernel = cv2.getTrackbarPos('Kernel', 'SVM Project1_3,4,5')
    gamma = cv2.getTrackbarPos('Gamma', 'SVM Project1_3,4,5') / 100.0
    C = cv2.getTrackbarPos('C', 'SVM Project1_3,4,5') / 100.0

    if kernel == kernel_options.index(cv2.ml.SVM_POLY):
        svm_model.setDegree(3)      # Poly 커널 차원(=양수) 설정
    svm_model.setKernel(kernel_options[kernel])
    svm_model.setGamma(gamma)
    svm_model.setC(C)
    svm_model.train(data, cv2.ml.ROW_SAMPLE, labels)

    # 전체 좌표 predict
    a = np.linspace(0, 639, 640)
    b = np.linspace(0, 639, 640)
    x, y = np.meshgrid(a,b)     # 2차원 그리드 생성
    cooldinate = np.c_[x.ravel(), y.ravel()].astype(np.float32)     # x좌표,y좌표를 가로방향으로 병합
    _, response = svm_model.predict(cooldinate)    # 예측
    response = response.reshape(x.shape)

    # 예측 결과 화면에 표시
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    for label in range(nClass):
        img[response == label] = back_colors[label]

    # 학습 데이터 표시
    for loc, label in zip(data, labels):
        cv2.circle(img, (int(loc[0]),int(loc[1])), 4, colors[label[0]], -1)

    # gamma, C, kernel값 화면에 출력
    cv2.putText(img,f'kernel: {kernel_name[kernel]}',(10,620), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img,f'C: {C}',(10,590), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(img,f'gamma: {gamma}',(10,560), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.imshow('SVM Project1_3,4,5', img)

# 트랙바 생성(단계 3,4,5)
cv2.namedWindow('SVM Project1_3,4,5')
cv2.createTrackbar('Gamma', 'SVM Project1_3,4,5', 1, 100, lambda n: trainMethod())    # 단계3
cv2.createTrackbar('C', 'SVM Project1_3,4,5', 1, 1000, lambda n: trainMethod())   # 단계4
cv2.createTrackbar('Kernel', 'SVM Project1_3,4,5', 0, len(kernel_options) - 1, lambda n: trainMethod())   # 단계5

trainMethod()   # 초기설정

plt.show()      # 단계1,2 plt show
cv2.waitKey()