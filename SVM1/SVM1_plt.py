# plt 이용. 3,4,5단계
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

nData = 100  # 각 클래스의 데이터 수
Range = 100  # 데이터의 전체 범위
nClass = 4  # 클래스 개수 (단계 2)
overlap = 10    # 클래스별 데이터 겹치는 범위

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
kernels = [cv2.ml.SVM_RBF, cv2.ml.SVM_LINEAR, cv2.ml.SVM_CHI2, cv2.ml.SVM_INTER, cv2.ml.SVM_SIGMOID, cv2.ml.SVM_POLY]
kernel_name = ['RBF','LINEAR','CHI2','INTER','SIGMOID','POLY']
def trainMethod():
    global svm_model, data, labels
    kernel = cv2.getTrackbarPos('Kernel', 'SVM Project1_3,4,5')
    gamma = cv2.getTrackbarPos('Gamma', 'SVM Project1_3,4,5') / 100.0
    C = cv2.getTrackbarPos('C', 'SVM Project1_3,4,5') / 100.0

    svm_model.setKernel(kernels[kernel])
    svm_model.setGamma(gamma)
    svm_model.setC(C)
    svm_model.train(data, cv2.ml.ROW_SAMPLE, labels)

    # 전체 좌표 predict
    a = np.linspace(-10, 100, 500)
    b = np.linspace(-10, 100, 500)
    x, y = np.meshgrid(a, b)  # 2차원 그리드 생성
    cooldinate = np.c_[x.ravel(), y.ravel()].astype(np.float32)  # x좌표,y좌표를 가로방향으로 병합
    _, response = svm_model.predict(cooldinate)  # 예측
    response = response.reshape(x.shape)

    # 예측 결과 화면에 표시
    ax.clear()
    ax.contourf(x, y, response,  alpha=0.5)
    ax.scatter(data[:, 0], data[:, 1], s=4 ** 2, c=labels)
    ax.set_title(f'gamma:{gamma}')
    ax.set_xlim([-10, 100])
    ax.set_ylim([-10, 100])

    plt.text(80, 0, f'gamma: {gamma}')
    plt.text(80, 5, f'C: {C}')
    plt.text(80, 10, f'kernel: {kernel_name[kernel]}')

    canvas.draw()
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    cv2.imshow('SVM Project1_3,4,5', img)




data, labels = generateData(nData, Range, nClass, overlap)

# SVM 설정
svm_model = cv2.ml.SVM_create()
svm_model.setType(cv2.ml.SVM_C_SVC)

# Matplotlib 설정
fig = plt.figure(figsize=(8, 6))
ax = plt.subplot(1,1,1)

canvas = FigureCanvas(fig)

cv2.namedWindow('SVM Project1_3,4,5')
cv2.createTrackbar('Gamma', 'SVM Project1_3,4,5', 1, 100, lambda n: trainMethod())    # 단계3
cv2.createTrackbar('C', 'SVM Project1_3,4,5', 1, 1000, lambda n: trainMethod())   # 단계4
cv2.createTrackbar('Kernel', 'SVM Project1_3,4,5', 0, len(kernels) - 1, lambda n: trainMethod())   # 단계5

trainMethod()   # 초기설정

cv2.waitKey()

