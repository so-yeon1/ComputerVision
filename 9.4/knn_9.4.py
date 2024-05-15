# knn 9.4 수정

# test data를 0~w*h까지 전 좌표로.(mashgird이용해 하나의 arr로제공?) : o
# nearest의 sample 전 좌표영역 넣고 해당 픽셀을 레이블에 맞게 색칠.    : o
# 결과 다르게 나온 sample에만 원 그리기                             :

# ??1) 결과가 반전인 것들만 원으로 표시? (반전: neighbor의 1위가 result로 채택x)
# ??2) 일부 테스트 데이터만 랜덤으로 몇 개 뽑아서 그 중에 반전인 것들만 원으로 표시?

# ================================================== #
## findNearest의 반환값 (N: 학습데이터 수) ##
# result: 레이블[N,1]
# neighbors: 가까운 거리의 레이블 k개 순차나열[N,k]
# dist: 테스트데이터와 가까운 학습데이터의 거리를 순차반환[N,k]
# ================================================== #

import numpy as np
import cv2
import matplotlib.pyplot as plt

# 시드 값 설정
np.random.seed(4222)

image_size = (200, 200, 3)  ############ 수정?
image = np.zeros(image_size, dtype=np.uint8)    ############ 수정? (변수만 넣는게 더 좋을 것 같음.)

# 학습 데이터 생성
L = 20
data = np.random.randint(0, 200, (L, 2)).astype(np.float32)

# 초기 레이블 생성 (불균형 데이터 생성)
class_distribution = [0.5, 0.5]  # 클래스 0과 클래스 1의 비율
labels = np.random.choice([0, 1], size=(L, 1), p=class_distribution).astype(np.float32) # 0,1 중에 무작위 추출. 0: 0.5, 1: 0.5확률로 추출

rand_test_data = np.random.randint(0, 200, (5, 2)).astype(np.float32)
# test_labels = np.random.choice([0, 1], size=(L, 1), p=class_distribution).astype(np.float32)    ########### 수정(test데이터의 레이블은 필요없음)

a = np.linspace(0, 199, 200)
b = np.linspace(0, 199, 200)
x, y = np.meshgrid(a, b)  # 2차원 그리드 생성
test_data = np.c_[x.ravel(), y.ravel()].astype(np.float32)  # x좌표,y좌표를 가로방향으로 병합
back_colors = [(229,209,92), (245,178,255)]

# k-NN 모델 생성
knn = cv2.ml.KNearest_create()

# 데이터 학습
knn.train(data, cv2.ml.ROW_SAMPLE, labels)

# 시각화 함수
def draw_image(k):
    # 그래프 초기화
    plt.figure(figsize=(8, 8))

    # 데이터 포인트 그리기
    for i in range(2):
        plt.scatter(data[labels.ravel()==i][:, 0], data[labels.ravel()==i][:, 1], 200, label=f'Class {i}', c=['b', 'r'][i])
    plt.scatter(rand_test_data[:, 0], rand_test_data[:, 1], 200, label='Test Data', c='g', marker='s')


    ret, results, neighbor, dist = knn.findNearest(test_data, k=k)
    results = results.reshape(x.shape)

    img = np.zeros((200, 200, 3), dtype=np.uint8)
    for label in range(2):
        img[results == label] = back_colors[label]


    # k 값에 따라 이웃을 고려하여 원 그리기
    for idx, point in enumerate(rand_test_data):  ############ 수정 (label은 필요x)
        label_text = idx  # 인덱스를 사용하여 0부터 오름차순으로 레이블 값 설정
        # if label == # 1)neighbors가 동점이 나왔을 경우:o  2) 만약 label과 neighbors의 1위가 같지 않으면 원 그리기:o  3) nearest함수 쓰지 말고 neighbor[인덱스]로 직접접근

        # plt.text(point[0], point[1], str(label_text), fontsize=12, color='black', ha='center', va='center') ######### 수정(중복. 하나는 없어도 됨)
        ret, results, neighbor, dist = knn.findNearest(point.reshape(1, -1), k=k)  ############## 수정(point.reshape(1,-1)으로 2차원배열으로 변경)
        print(neighbor[0][0])
        print(results)
        unique, cnt = np.unique(neighbor, return_counts=True)
        max_cnt = np.max(cnt)
        print(dist[0])
        if (cnt == max_cnt).sum() > 1 and (neighbor[0][0] != results):      # 동점인 경우
            # radius = np.mean(dist[0]) ** 0.5
            radius = np.sqrt(dist[0, k - 1])  ########### 수정(mean을 쓴 이유가 있는지?)
            # 빨간색에 가까우면 검정 파란색에 가까우면 흰색
            circle_color = 'b' if results[0][0] == 0 else 'r'
            plt.text(point[0], point[1], str(label_text), fontsize=20, color=circle_color)
            circle = plt.Circle((point[0], point[1]), radius, color=circle_color, fill=False)
            plt.gca().add_patch(circle)

    rgbimg = img[:, :, ::-1]
    plt.imshow(rgbimg)
    plt.title(f'KNN Visualization (k={k})')
    plt.legend()
    plt.grid(True)
    # plt.axis('equal')             ############ 수정
    plt.show()

    # 정확도 계산
    ret, result, neighbors, dist = knn.findNearest(data, k=k)
    accuracy = np.mean(result == labels)
    print(f'Accuracy: {accuracy:#.2f}'+"%" " " f'k :  {k}')


# # 트랙바 콜백 함수       ######### 수정(굳이 없어도 됨. createTrackbar에서 바로 draw_image 호출)
# def on_trackbar(val):
#     k = cv2.getTrackbarPos('k', 'KNN Visualization')
#     draw_image(k)

# 트랙바 생성
cv2.namedWindow('KNN Visualization')
cv2.createTrackbar('k', 'KNN Visualization', 1, 5, draw_image)

# 초기 이미지 표시
draw_image(1)       ############### 수정(위치)

cv2.waitKey(0)
cv2.destroyAllWindows()