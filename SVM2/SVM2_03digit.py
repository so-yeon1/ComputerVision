"""
Handwritten digits recognition using SVM and HoG features and varying the number of
training/testing images with pre-processing of the images. A grid-search on C and gamma is also carried out.

KNN에서 실험했던 것처럼 deskew 보정과 hog 추출한 결과를 가지고 SVM으로 분류 처리를 시행하는데
C, gamma를 바꿀 때마다 어떻게 SVM의 성능이 달라지는지 비교해 본다.
사전에 학습에 사용될 비율을 설정할 수 있다.

참고:
    이 프로그램은 아래 프로그램에서는 분류과정을 KNN으로 했던 것을 SVM으로 대치한 점이 다르다.
    2_4_knn_handwritten_digits_recognition_k_training_testing_preprocessing_hog.py
    위의 프로그램은 KNN 실행전에 deskew 보정과 hog 추출을 수행한다.
    KNN과 SVM의 분류 성능을 비교해 보는데 의의가 있다.

주의:
    이 프로그램의 수행을 위해서는 현재 폴더 위의 data 폴더에 다음 파일이 준비되어 있어야 한다.
        '../data/digits.png'



"""

# Import required packages:
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Constants:
SIZE_IMAGE = 20
NUMBER_CLASSES = 10


def load_digits_and_labels(big_image):
    """ Returns all the digits from the 'big' image and creates the corresponding labels for each image"""

    # Load the 'big' image containing all the digits:
    digits_img = cv2.imread(big_image, 0)

    # Get all the digit images from the 'big' image:
    number_rows = digits_img.shape[1] / SIZE_IMAGE
    rows = np.vsplit(digits_img, digits_img.shape[0] / SIZE_IMAGE)

    digits = []
    for row in rows:
        row_cells = np.hsplit(row, number_rows)
        for digit in row_cells:
            digits.append(digit)
    digits = np.array(digits)

    # Create the labels for each image:
    labels = np.repeat(np.arange(NUMBER_CLASSES), len(digits) / NUMBER_CLASSES)
    return digits, labels


def deskew(img):
    """Pre-processing of the images"""

    m = cv2.moments(img)
    if abs(m['mu02']) < 1e-2:
        return img.copy()
    skew = m['mu11'] / m['mu02']
    M = np.float32([[1, skew, -0.5 * SIZE_IMAGE * skew], [0, 1, 0]])
    img = cv2.warpAffine(img, M, (SIZE_IMAGE, SIZE_IMAGE), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
    return img


def svm_init(C=12.5, gamma=0.50625):
    """Creates empty model and assigns main parameters"""

    model = cv2.ml.SVM_create()
    model.setGamma(gamma)
    model.setC(C)
    model.setKernel(cv2.ml.SVM_RBF)
    model.setType(cv2.ml.SVM_C_SVC)
    model.setTermCriteria((cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))

    return model


def svm_train(model, samples, responses):
    """Returns the trained SVM model based on the samples and responses"""

    model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    return model


def svm_predict(model, samples):
    """Returns the predictions"""

    return model.predict(samples)[1].ravel()


def svm_evaluate(model, samples, labels):
    """Returns SVM evaluation (accuracy)"""

    predictions = svm_predict(model, samples)
    accuracy = (labels == predictions).mean()
    # print('Percentage Accuracy: %.2f %%' % (accuracy * 100))
    return accuracy * 100


def get_hog():
    """ Get hog descriptor """

    # cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType,
    # L2HysThreshold, gammaCorrection, nlevels, signedGradient)
    hog = cv2.HOGDescriptor((SIZE_IMAGE, SIZE_IMAGE), (8, 8), (4, 4), (8, 8), 9, 1, -1, 0, 0.2, 1, 64, True)

    print("hog descriptor size={}".format(hog.getDescriptorSize()))

    return hog


def raw_pixels(img):
    """Return raw pixels as feature from the image"""

    return img.flatten()


def sort_accuracy(accuracy_list):   # C, gamma, accuracy
    # 정확고가 높은 것부터 낮은 순으로 소팅한다.
    # lambda 인자는 어차피 dummy parameter이므로 아무 문자나 써도 상관 없다.
    # 2번 인자에 기반해서 소팅한다.
    accuracy_list2 = sorted(accuracy_list, key=lambda _: _[2], reverse=True)
    return accuracy_list2



# Load all the digits and the corresponding labels:
digits, labels = load_digits_and_labels('digits.png')

# Shuffle data
# Constructs a random number generator:
rand = np.random.RandomState(1234)
# Randomly permute the sequence:
shuffle = rand.permutation(len(digits))
digits, labels = digits[shuffle], labels[shuffle]

# HoG feature descriptor:
hog = get_hog()

# Compute the descriptors for all the images.
# In this case, the HoG descriptor is calculated
hog_descriptors = []
for img in digits:
    hog_descriptors.append(hog.compute(deskew(img)))
hog_descriptors = np.squeeze(hog_descriptors)

# At this point we split the data into training and testing:
percnt_train = 0.9  # 학습 데이터의 비율. 나머지(1-percnt_train)는 테스팅 데이터의 비율. 0.9-> 90%
#percnt_train = 0.5  # 학습 데이터의 비율. 나머지(1-percnt_train)는 테스팅 데이터의 비율.
partition = int(percnt_train * len(hog_descriptors))
hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [partition])
labels_train, labels_test = np.split(labels, [partition])
print(f"percentage of train data={percnt_train*100:#.1f}%,"
      f"\nnumber of training data={partition}, number of testing data={len(hog_descriptors)-partition}")
print(f"hog_descriptors_train.shape={hog_descriptors_train.shape}, hog_descriptors_test.shape={hog_descriptors_test.shape}")
print(f"labels_train.shape={labels_train.shape}, labels_test.shape={labels_test.shape}")



print('\nTraining SVM model ...')
# Create a dictionary to store the accuracy when testing:
results = defaultdict(list)

# SVM_03 취지에 맞는 설정
# gamma_list = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]   # 추천.
# C_list = [0.1, 0.3, 0.5, 0.9, 1.3, 5, 12.5]      # 추천.

# SVM_02 예제와 비교해 보기 위한 설정, 1회 검증용
gamma_list = [0.1, 0.3, 0.50625, 0.7, 0.9, 1.1, 1.3, 1.5]
C_list = [12.5]

accuracy_list = []
for C in C_list:       # 수정 필요: 같은 곳에 여러번 그리면 구분이 않됨.
    for gamma in gamma_list:
        model = svm_init(C, gamma)
        svm_train(model, hog_descriptors_train, labels_train)
        acc = svm_evaluate(model, hog_descriptors_test, labels_test)
        print(f"C={C:4.1f}, gamma={gamma:3.1f}: accuracy={acc:5.2f}")
        accuracy_list.append((C, gamma, acc))
        results[C].append(acc)

print("\n정확도 순으로 소팅해서 출력한 결과입니다.")
accuracy_list2 = sort_accuracy(accuracy_list)
for C, gamma, acc in accuracy_list2:
    print(f"C={C:4.1f}, gamma={gamma:3.1f}: accuracy={acc:5.2f}")

# Create the dimensions of the figure and set title:
fig = plt.figure(figsize=(10, 6))
plt.suptitle(f"SVM handwritten digits recognition(training data={percnt_train*100}%)", fontsize=14, fontweight='bold')
#fig.patch.set_facecolor('silver')

# Show all results using matplotlib capabilities:
ax = plt.subplot(1, 1, 1)
ax.set_xlim(0, 1.5)
#dim = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5]

for key in results:
    ax.plot(gamma_list, results[key], linestyle='--', marker='o', label=str(key))

#plt.legend(loc='upper left', title="C")
plt.legend(title="C")
plt.title(f'Accuracy of the SVM model varying both C and gamma')
plt.xlabel("gamma")
plt.ylabel("accuracy")
plt.show()
