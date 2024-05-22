import numpy as np
import cv2
import matplotlib.pyplot as plt

# 시드 값 설정
np.random.seed(51)

image_size = (200, 200, 3)
image = np.zeros(image_size, dtype=np.uint8)

# 중복 없는 학습 데이터 생성
L = 20
data = np.random.randint(0, 200, (L, 2)).astype(np.float32)
data = np.unique(data, axis=0)

while len(data) < L:
    newData = np.random.randint(0,200,(L-len(data),2)).astype(np.float32)
    data = np.append(data, newData, axis=0)
    data = np.unique(data, axis=0)

# 초기 레이블 생성 (불균형 데이터 생성)
class_distribution = [0.2, 0.8]
labels = np.random.choice([0, 1], size=(L, 1), p=class_distribution).astype(np.float32)

# 원 표시용 랜덤 테스트 데이터 생성
rand_test_data = np.random.randint(0, 200, (25, 2)).astype(np.float32)

# 테스트 데이터(2차원 그리드) 생성
a = np.linspace(0, 199, 200)
b = np.linspace(0, 199, 200)
x, y = np.meshgrid(a, b)
test_data = np.c_[x.ravel(), y.ravel()].astype(np.float32)

# 배경색
back_colors = [(229,209,92), (245,178,255)]

# knn 모델 생성
knn = cv2.ml.KNearest_create()

# knn 학습
knn.train(data, cv2.ml.ROW_SAMPLE, labels)

# 시각화 함수
def draw_image(k):
    # 그래프 초기화
    plt.figure(figsize=(8, 8))

    # 데이터 포인트 그리기
    for i in range(2):
        plt.scatter(data[labels.ravel()==i][:, 0], data[labels.ravel()==i][:, 1], 200, label=f'Class {i}', c=['b', 'r'][i])
    plt.scatter(rand_test_data[:, 0], rand_test_data[:, 1], 200, label='Test Data', c='g', marker='s')

    # knn 분류 test
    ret, results, neighbor, dist = knn.findNearest(test_data, k=k)

    # 학습 결과에 따라 전 영역 coloring
    img = np.zeros((200, 200, 3), dtype=np.uint8)
    color_results = results.reshape(x.shape)
    for label in range(2):
        img[color_results == label] = back_colors[label]


    # 결과가 반전인 테스트 데이터에 원 그리기
    for idx, point in enumerate(rand_test_data):
        label_text = idx
        ret, results, neighbor, dist = knn.findNearest(point.reshape(1, -1), k=k)

        # neighbor가 동점인 데이터 탐색
        unique, cnt = np.unique(neighbor, return_counts=True)
        max_cnt = np.max(cnt)

        # neighbor가 동점이고, neighbor의 1위와 results가 같지 않을 경우에만 원 그리기
        if (cnt == max_cnt).sum() > 1 and (neighbor[0][0] != results[0][0]):
            radius = np.sqrt(dist[0, k - 1])
            circle_color = 'b' if results[0][0] == 0 else 'r'
            plt.text(point[0], point[1], str(label_text), fontsize=20, color=circle_color)
            circle = plt.Circle((point[0], point[1]), radius, color=circle_color, fill=False)
            plt.gca().add_patch(circle)
            print(idx, neighbor, results)

    rgbimg = img[:, :, ::-1]
    plt.imshow(rgbimg)
    plt.title(f'KNN Visualization (k={k})')
    plt.legend()
    plt.grid(True)
    plt.show()

# 트랙바 생성
cv2.namedWindow('KNN Visualization')
cv2.createTrackbar('k', 'KNN Visualization', 1, 5, draw_image)

# 초기 이미지 표시
draw_image(1)

cv2.waitKey(0)
cv2.destroyAllWindows()