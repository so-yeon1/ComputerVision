"""

제목: knn 학습데이터에 대한 분류 성공률 환산 프로그램

이 프로그램의 정확도 측정 방법에는 오류가 있는 듯함.
2_0_knn_handwritten_digits_recognition_introduction_b.py에서 사용하였던 get_accuracy로 계산하면
항생 50%정도만 나온다.


개요:
    본 프로그램은 k-Nearest Neighbour의 학습 모델을 구축했을 때
    학습 모델을 만드는데 기여했던 학습 데이터들을 knn 모델이 k 값에 따라 어떤 분류 결과를 내는지 검증해 보고자 한다.

절차:
    단계 1) 임의로 생성한 L개의 (x,y) 좌표 데이터를 임의로 2개의 그룹으로 나누어 학습 데이터를 생성한다.
       랜덤하게 배정된 labels를 이용하여 학습 데이터를 생성하였다.
    단계 2) knn 학습 모델을 구축한다.
    단계 3) 다양한 k값의 선정에 대해 학습된 데이터는 잘 맞추는지 확인한다.
        주목할 사항(특히 dist)
        result = 주어진 테스트 입력에 대해 분류한 결과의 레이블 정보
        neighbors = k만큼의 후보군들의 레이블 정보.
                    첫 번째는 result와 같다. 그 이후는 차선 후보들의 레이블이다.
                    k개 만큼의 후보의 수를 갖고 있다.
        dist = 각 k개 후보들의 유클리디안 거리이다.
               현재 학습데이터를 테스트 데이터로 사용하기 때문에 첫번째 후보의 거리는 0이다.


실험 결과
    40만개의 2차원 랜덤 데이터를 K-nn 모델로 학습시켜 본 결과 학습데이터에 대해 불과 51.25%의 정확도를 달성하였다.
        아래는 출력문 내용
            test=train: L=400000, k=1: Accuracy=51.25%
            test=train: L=40000, k=1: Accuracy=62.23%
            test=train: L=4000, k=1: Accuracy=91.50%
            test=train: L=400, k=1: Accuracy=99.00%
            test=train: L=100, k=1: Accuracy=100.00%


    잘 공개된 것은 아니지만 확실한 것은 모든 데이터를 그대로 저장하고 있는 것은 아닌 것은 확실하다.
    만약 generalization을 행하지 않고 모든 데이터를 그대로 저장한다면 100% 정확도 달성을 이룰 것이다.
    => 저장하는 방식으로 코딩을 고쳐보자: cs231n 코드 참조.

    차후 이런 성질의 랜덤 데이터를 DNN에 적용하여 학습데이터에 대한 정확도를 얼마까지 기대할 수 있는지 살펴보자.


미션
    현재 이 프로그램은 L개의 2차원 데이터에 대해 k값의 변화에 따른 학습 정확도를 측정하게 되어 있다.
    이 프로그램을 기반으로 L개의 D차원 학습데이터에 대한 k=1의 학습정확도를 그래픽으로 출력하시오.
    - x축은 L, y축은 D, z축은 Accuracy.
    - L=[100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200, 102400, 204800, 409600]
    - L=[100, 400, 1600, 6400, 25600, 102400, 409600]
    - L=[100, 1600, 25600, 409600]
    - D = [2, 4, 8, 16, 32, 64, 128]

    같은 데이터에 대해 다른 분류 알고리즘과의 성능을 비교하고자 한다.
    대상 알고리즘 ANN, DNN



검토 1 - knn 모델의 크기는? 학습 모델을 저장할 수는 없나?
    현재 상황: scikit-learn에는 iris의 모든 데이터 세트가 저장된 세트를 load 할 수 있다.
        http://blog.naver.com/PostView.nhn?blogId=owl6615&logNo=221457206097&parentCategoryNo=118&categoryNo=119&viewDate=&isShowPopularPosts=false&from=postView
        그러나 이는 아직 모델을 저장하는 기능이 있다는 것을 뜻하는 것은 아니다.
    본 코딩에는 sys.getsizeof() 함수로 knn 모델의 크기를 학습 전과 학습 후로 나누어 판단해 보려 하였으나
    둘 다 같은 크기로 결과가 나와 모델 객체가 지정하는 학습 데이터는 포함하지 않는 것으로 판명되었다.

    knn 객체를 pickle() 함수로 통째로 파일로 저장하여 간접적으로 크기 변화를 관찰하고자 하였으나,
    pickle() 함수가 knn 객체의 저장은 실행히 못하고 오류로 중단되었다.
    현재 이 부분은 주석문으로 처리하였다.

검토 2 - 본 예제에서는 findNearest() 함수를 호출한 결과를 일일이 비교해서 accuracy를 판단했는데
        predict() 함수로 한꺼번에 일을 처리할 수는 없을까?
        => 불가능한 것으로 보인다. predict() 함수에 k를 전달할 수 있는 장치가 없다.
           test데이터와 flag만 넣는다.



retval, results, neighborResponses, dist = cv.ml_KNearest.findNearest( samples, k[, results[, neighborResponses[, dist]]] )
    samples: Input samples stored by rows. It is a single-precision floating-point matrix of <number_of_samples> * k size.
    k: Number of used nearest neighbors. Should be greater than 1.
    results: Vector with results of prediction (regression or classification) for each input sample.
            It is a single-precision floating-point vector with <number_of_samples> elements.
    neighborResponses: Optional output values for corresponding neighbors.
            It is a single- precision floating-point matrix of <number_of_samples> * k size.
    dist: Optional output distances from the input vectors to the corresponding neighbors.
        It is a single-precision floating-point matrix of <number_of_samples> * k size.


"""

# Import required packages:
import cv2
import numpy as np
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------------
# 단계 1: 학습 데이터를 만든다. 랜덤한 값(x, y)으로 데이터를 L개 생성한다.
# 이들을 2개의 그룹으로 랜덤하게 나눈다.
# --------------------------------------------------------------------------------------------
# The data is composed of L points: [0, 100)의 범위를 가진 L개의 점(2차원: x, y) 데이터 정의
# train data로 쓰인다.
print('\nStep 1: Making random training data & labels..')
L = 400       # 작은 수를 사용하면 프린트하기 수월하다.
data = np.random.randint(0, 100, (L, 2)).astype(np.float32)
print('data for training:', type(data), data.shape)

# We create the labels (0: red, 1: blue) for each of the L points:
# 각 점에 대해 0 혹은 1의 레이블을 할당. train을 위한 label로 쓰인다.
labels = np.random.randint(0, 2, (L, 1)).astype(np.float32)
print('Labels for training:', type(labels), labels.shape)


# --------------------------------------------------------------------------------------------
# 단계 2: knn 학습 모델을 구축한다.
# --------------------------------------------------------------------------------------------
# k-NN creation:
print('\nStep 2: Creating & training a knn model..')
knn = cv2.ml.KNearest_create()

import time

s_time = time.time()
# k-NN training: 학습용 데이터, data와 이를 지도학습 시킬 수 있는 label이 필요하다.
knn.train(data, cv2.ml.ROW_SAMPLE, labels)
print(f'training time={time.time()-s_time:#.2f}')

"""
# knn의 학습 전후의 크기를 가늠하고자 그 객체의 학습 전후 크기를 출력해 본다. => 동일
# pickle() 함수로 파일로 저장한 후 파일 크기를 읽는 방법으로 간접적으로 추정 -> 오류 발생으로 실패 
import sys
import pickle
from os.path import getsize
# 메모리 양의 변화를 추정하려 하였으나 무의미한 시도였다.
print(f'model size before training= {sys.getsizeof(knn)}')
#print(f'model size after training= {sys.getsizeof(knn)}')
# 링크가 걸린 실제 메모리를 포함한 크기를 반환하지는 않았다.

# 객체를 파일에 저장하기
with open("tmp_knn_before.bin", "wb") as file:
    pickle.dump(knn, file)
f_size = getsize("tmp_knn_before.bin")
print(f'knn file size before training={f_size}')

# 객체를 파일에서 불러오기
#with open("tmp_object.bin", "rb") as file:
#    img2 = pickle.load(file)

with open("tmp_knn_after.bin", "wb") as file:
    pickle.dump(knn, file)
f_size = getsize("tmp_knn_after.bin")
print(f'knn file size before training={f_size}')

exit(0)

"""


# --------------------------------------------------------------------------------------------
# 단계 3: 다양한 k값의 변경에 대해 학습된 데이터는 잘 맞추는지 확인한다.
# 관심 사항(특히 dist)
#   result = 주어진 테스트 입력에 대해 분류한 결과의 레이블 정보
#   neighbors = k만큼의 후보군들의 레이블 정보.
#               첫 번째는 result와 같다. 그 이후는 차선 후보들의 레이블이다.
#               k개 만큼의 후보의 수를 갖고 있다.
#   dist = 각 k개 후보들의 유클리디안 거리이다.
#           현재 학습데이터를 테스트 데이터로 사용하기 때문에 첫번째 후보의 거리는 0이다.
# --------------------------------------------------------------------------------------------
print('\nStep 3: Checking the knn model according to k selection..')
from collections import Counter

K = [1, 3, 5]
K = [1]
#K = [1, 3]

def print_ret_values(ret_val, rslt_v='no', nfg_v='no', dst_v='no'):
    """findNearest() 함수가 반환하는 반환 값을 이해하기 위해 반환값을 출력하는 함수"""
    retval, results, neighbours, dist = ret_val
    #print('retval:', type(retval), retval)   # 모드 k 에 대해 0이 나옴. 삭제..
    print('results:', type(results), results.shape)
    if rslt_v != 'no': print(results)
    print('neighbours:', type(neighbours), neighbours.shape)
    if nfg_v != 'no': print(neighbours)
    print('dist:', type(dist), dist.shape)
    if dst_v != 'no': print(dist)

def get_accuracy(predictions, labels):
    """Returns the accuracy based on the coincidences between predictions and labels"""
    # 본 정확도는 test 데이터에 대해서만 환산됨에 유의...
    accuracy = (np.squeeze(predictions) == labels).mean()   # 'mean()' returns the average of the array elements.
    return accuracy * 100   # labels가 10개라서 맞힌 갯수를 평균(mean)을 구해서 100을 곱하면 정확도가 나온다.

for k in K:
    print(f'\ntest=train: k={k}, num of test data={len(data)} -------')
    s_time = time.time()
    ret_val = knn.findNearest(data, k)
    e_time = time.time()
    print(f'testing time: whole={e_time - s_time:#.2f}, unit={(e_time - s_time)/len(data):#.2f}')
    #print_ret_values(ret_val, rslt_v='yes', nfg_v='yes', dst_v='yes')
    #print_ret_values(ret_val, 'yes', 'yes', 'yes')     # 타입, shape와 값 모두 출력한다.
    print_ret_values(ret_val)   # 값은 출력하지 않는다. shape, type만 출력한다.

    ret, results, neighbours, dist = ret_val

    cmp = labels == results  # 학습데이터의 label과 knn으로 판단해서 분류한 label(results)이 같은지 비교한다.
    #print('cmp:', type(cmp), cmp.dtype, cmp.shape, '\n', cmp)
    cmp_f = cmp.flatten()
    #print('cmp.f:', type(cmp_f), cmp_f.dtype, cmp_f.shape, '\n', cmp_f)
    dict = Counter(cmp_f)
    #print(f'dict={dict}')
    print(f'test=train: L={L}, k={k}: Accuracy={dict[True] * 100 / len(cmp):#.2f}%')

    # 보류: 다른 방법으로 정확도를 계산해보려 했는데 안됨. 현재 어디가 오류인지 모르겠음..
    # 다른 예제에서 사용하였던 함수로 accuracy를 구해본다.
    acc = get_accuracy(results, labels)
    print(f"k={k}: Accuracy2={acc:#6.2f}")



exit(0)
