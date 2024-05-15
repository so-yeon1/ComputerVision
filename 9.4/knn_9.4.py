import numpy as np
import cv2
import matplotlib.pyplot as plt

# 시드 값 설정
np.random.seed(42)

image_size = (200, 200)
image = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

# 데이터 생성
L = 10
data = np.random.randint(0, 100, (L, 2)).astype(np.float32)

# 초기 레이블 생성 (불균형 데이터 생성)
class_distribution = [0.5, 0.5]  # 클래스 0과 클래스 1의 비율
labels = np.random.choice([0, 1], size=(L, 1), p=class_distribution).astype(np.float32)

test_data = np.random.randint(0, 100, (L, 2)).astype(np.float32)
test_labels = np.random.choice([0, 1], size=(L, 1), p=class_distribution).astype(np.float32)

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
        plt.scatter(data[labels.ravel()==i][:, 0], data[labels.ravel()==i][:, 1], label=f'Class {i}', c=['b', 'r'][i])
    plt.scatter(test_data[:, 0], test_data[:, 1], label='Test Data', c='g')

    # k 값에 따라 이웃을 고려하여 원 그리기
    for idx, (point, label) in enumerate(zip(test_data, test_labels)):
        label_text = idx  # 인덱스를 사용하여 0부터 오름차순으로 레이블 값 설정
        plt.text(point[0], point[1], str(label_text), fontsize=12, color='black', ha='center', va='center')

        ret, results, neighbor, dist = knn.findNearest(np.array([point], dtype=np.float32), k=k)
        radius = np.mean(dist[0]) ** 0.5
        # 빨간색에 가까우면 검정 파란색에 가까우면 흰색
        circle_color = 'b' if results[0][0] == 0 else 'r'
        plt.text(point[0], point[1], str(label_text), fontsize=12, color=circle_color, ha='center', va='center')
        circle = plt.Circle((point[0], point[1]), radius, color=circle_color, fill=False)
        plt.gca().add_patch(circle)

    """빨간 배경과 파란 배경"""
    # 각 픽셀에 대해 k-NN 모델을 사용하여 색을 할당
    # for i in range(image_size[0]):
    #     for j in range(image_size[1]):
    #         # 이미지를 반반씩 나눠서 빨간색과 파란색으로 채우기
    #         if i < image_size[0] // 2:
    #             color = (0, 0, 255)  # 빨간색
    #         else:
    #             color = (255, 0, 0)  # 파란색
    #
    #         image[i, j] = color

    # 이미지 표시
    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.axis('off')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'KNN Visualization (k={k})')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

    # 정확도 계산
    ret, result, neighbors, dist = knn.findNearest(data, k=k)
    accuracy = np.mean(result == labels)
    print(f'Accuracy: {accuracy:#.2f}'+"%" " " f'k :  {k}')

# 초기 이미지 표시
draw_image(1)

# 트랙바 콜백 함수
def on_trackbar(val):
    k = cv2.getTrackbarPos('k', 'KNN Visualization')
    draw_image(k)

# 트랙바 생성
cv2.namedWindow('KNN Visualization')
cv2.createTrackbar('k', 'KNN Visualization', 1, 10, on_trackbar)

cv2.waitKey(0)
cv2.destroyAllWindows()