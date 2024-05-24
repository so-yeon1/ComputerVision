'''
4조_박소연, 강희정, 이태훈

* 프로그램 목적:
1) gamma에 따른 SVM 분류 특성을 보여준다.
2) 트랙바를 이용해 gamma, C, kernel함수에 따른 SVM 분류 특성을 보여준다.

* 작동 방법:
[단계 1, 2]
 1) 클래스의 수를 설정하고, 학습 데이터를 생성한다.
 2) SVM 모델 생성 및 학습을 수행한다.
 3) 테스트데이터(전체 좌표)를 하나의 numpy 배열로 생성한다.
 4) SVM 예측을 수행한다.
 5) 예측 결과를 그래프에 표시한다.
    -> contourf(x,y,z) : 등고선 함수. z값에 따라 다른 색으로 영역 표시함
        x,y : z값의 좌표
          z : 높이값. (N,M)형태의 배열
 6) gamma값을 달리하여 2 ~ 5의 과정을 반복해 여러개의 그림을 그린다.
    -> (gamma값들이 들어있는 배열을 만들고, 반복문을 통해 각각의 gamma값에 차례로 접근한다.)

[단계 3, 4, 5]
 1) 클래스의 수를 설정하고, 학습 데이터를 생성한다.
 2) SVM 모델을 생성한다.
 3) gamma, C, kernel함수를 제어할 트랙바를 각각 생성한다.
    -> cv2.createTrackbar()
 4) 트랙바 콜백 함수(trainMethod())에서 트랙바로부터 전달받은 gamma, C, kernel함수 값을 설정한다.
 5) SVM 학습을 수행한다.
 6) [단계 1, 2]의 3 ~ 5와 같은 동작을 수행한다.
 7) 트랙바 값 변경시마다, 업데이트된 값에 따라 분류 결과의 변화를 보인다.
    -> canvas.draw() : 업데이트된 figure 객체를 그림
       img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8') : figure객체를 rgb 픽셀값을 가진 numpy 배열로 변환
       img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,)) : 이미지를 imshow할 수 있는 형식으로 reshape
       cv2.imshow('SVM1_(3, 4, 5)', img) : 이미지 표시

=> 단계 1,2는 line 44~ 103에
   단계 3,4,5는 line 108~182에 작성함.
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# ====================================== 단계 1,2 (line 44~103)======================================= #
nClass = 4  # 클래스 개수 (단계 2)
L = 100  # 각 클래스의 데이터 수
R = 100  # 데이터의 전체 범위
overlap = 20    # 클래스별 데이터 겹치는 범위
gammas = [0.01, 0.1, 0.5, 1]     # 감마값

## 데이터 및 레이블 생성
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


# 데이터 생성 및 레이블 배정
data, labels = generateData(L, R, nClass, overlap)

# 그래프 창 생성
fig, axes = plt.subplots(1, len(gammas),figsize=(15, 3))
fig.canvas.manager.set_window_title('SVM1_(1, 2)')
plt.suptitle(f"class: {nClass}", fontsize=14,
             fontweight='bold')
plt.subplots_adjust(top=0.8)

# gamma값에 따른 SVM 학습 및 시각화
for gamma, ax in zip(gammas, axes):
    # SVM 모델 생성 및 설정
    svm_model = cv2.ml.SVM_create()
    svm_model.setKernel(cv2.ml.SVM_RBF)
    svm_model.setType(cv2.ml.SVM_C_SVC)
    svm_model.setGamma(gamma)   # gamma 값 설정
    svm_model.setC(10)

    # SVM 모델 학습
    svm_model.train(data, cv2.ml.ROW_SAMPLE, labels)

    # 전체 좌표(테스트데이터)를 하나의 numpy 배열로 생성
    a = np.linspace(-5, 105, 300)
    b = np.linspace(-5, 105, 300)
    x, y = np.meshgrid(a, b)  # 2차원 그리드 생성
    test_data = np.c_[x.ravel(), y.ravel()].astype(np.float32)  # 전체좌표를 (x,y)형태의 numpy배열로 생성

    # 전 영역을 한꺼번에 predict
    _, response = svm_model.predict(test_data)
    response = response.reshape(x.shape)    # 결과를 화면에 표시하기 위해 response와 x,y의 shape 일치시키기

    # 예측 결과 화면에 표시
    ax.contourf(x, y, response, alpha=0.5)

    # 학습데이터 표시
    ax.scatter(data[:, 0], data[:, 1], s=5**2, c=labels, edgecolor='k')
    ax.set_title(f'gamma:{gamma}')
plt.show()


# ================================ 단계 3,4,5 (line 108~182) ===================================== #

nClass = 3  # 클래스 개수
L = 100  # 각 클래스의 데이터 수
R = 100  # 데이터의 전체 범위
overlap = 25    # 겹치는 범위

# 데이터 생성 및 레이블 배정
data, labels = generateData(L, R, nClass, overlap)

# SVM 모델 생성
svm_model = cv2.ml.SVM_create()
svm_model.setType(cv2.ml.SVM_C_SVC)

kernels = [cv2.ml.SVM_RBF, cv2.ml.SVM_LINEAR, cv2.ml.SVM_CHI2, cv2.ml.SVM_INTER, cv2.ml.SVM_SIGMOID, cv2.ml.SVM_POLY]
kernel_name = ['RBF','LINEAR','CHI2','INTER','SIGMOID','POLY']

## SVM 학습 및 예측(트랙바 콜백 함수)
def trainMethod():
    # 트랙바로부터 값 받아오기
    kernel = cv2.getTrackbarPos('Kernel', 'SVM1_(3, 4, 5)')
    gamma = cv2.getTrackbarPos('Gamma', 'SVM1_(3, 4, 5)') / 100.0
    C = cv2.getTrackbarPos('C', 'SVM1_(3, 4, 5)') / 10.0

    # SVM 모델 설정
    if kernel == kernels.index(cv2.ml.SVM_POLY):
        svm_model.setDegree(3)      # POLY 커널 차원 설정
    svm_model.setKernel(kernels[kernel])
    svm_model.setGamma(gamma)
    svm_model.setC(C)

    # SVM 모델 학습
    svm_model.train(data, cv2.ml.ROW_SAMPLE, labels)

    # 전체 좌표(테스트데이터)를 하나의 numpy 배열로 생성
    a = np.linspace(-5, 105, 300)
    b = np.linspace(-5, 105, 300)
    x, y = np.meshgrid(a, b)  # 2차원 그리드 생성
    cooldinate = np.c_[x.ravel(), y.ravel()].astype(np.float32)  # 전체좌표를 (x,y)형태의 numpy배열로 생성

    # 전 영역을 한꺼번에 predict
    _, response = svm_model.predict(cooldinate)
    response = response.reshape(x.shape)    # 결과를 화면에 표시하기 위해 response와 x,y의 shape 일치시키기

    # 예측 결과 화면에 표시
    ax.clear()
    ax.contourf(x, y, response,  alpha=0.5)

    # 학습 데이터 표시
    ax.scatter(data[:, 0], data[:, 1], s=9**2, c=labels, edgecolor='k')

    plt.text(75, 0, f'gamma: {gamma}',size=14)
    plt.text(75, 7, f'C: {C}',size=14)
    plt.text(75, 14, f'kernel: {kernel_name[kernel]}',size=14)

    # 업데이트된 예측 결과 시각화
    canvas.draw()   # figure 객체를 그림(트랙바 호출시마다)
    img = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')   # figure객체를 rgb 픽셀값을 가진 numpy 배열로 변환
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))   # 이미지를 표시할 수 있는 형식으로 만들어줌
    cv2.imshow('SVM1_(3, 4, 5)', img)


fig = plt.figure(figsize=(8, 6))
ax = plt.subplot(1,1,1)
plt.suptitle(f"class: {nClass}", fontsize=14)

canvas = FigureCanvas(fig)  # 그릴 figure 객체

# 트랙바 생성
cv2.namedWindow('SVM1_(3, 4, 5)')
cv2.createTrackbar('Gamma', 'SVM1_(3, 4, 5)', 1, 100, lambda n: trainMethod())    # 단계3
cv2.createTrackbar('C', 'SVM1_(3, 4, 5)', 1, 10000, lambda n: trainMethod())   # 단계4
cv2.createTrackbar('Kernel', 'SVM1_(3, 4, 5)', 0, len(kernels) - 1, lambda n: trainMethod())   # 단계5

trainMethod()   # 초기설정

cv2.waitKey()