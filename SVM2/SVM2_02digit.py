"""
Handwritten digits recognition using SVM and HoG features and varying the number of
training/testing images with pre-processing(deskew & hog) of the images
KNN에서 실험했던 것처럼 deskew 보정과 hog 추출한 결과를 가지고 SVM으로 분류 처리를 시행해 본다.

참고:
    이 프로그램은 아래 프로그램에서는 분류과정을 KNN으로 했던 것을 SVM으로 대치한 점이 다르다.
    2_4_knn_handwritten_digits_recognition_k_training_testing_preprocessing_hog.py
    위의 프로그램은 KNN 실행전에 deskew 보정과 hog 추출을 수행한다.

주의:
    이 프로그램의 수행을 위해서는 현재 폴더 위의 data 폴더에 다음 파일이 준비되어 있어야 한다.
        '../data/digits.png'

hog descriptor size: '144'

Training KNN(skew correction & Hog) model using 500 training data, which is 10.00% ...
4500: k:Acc => 1:95.9 2:95.6 3:96.0 4:96.1 5:96.0 6:95.8 7:95.8 8:95.7 9:95.8

Training KNN(skew correction & Hog) model using 1000 training data, which is 20.00% ...
4000: k:Acc => 1:96.7 2:96.7 3:97.1 4:97.4 5:97.0 6:97.1 7:96.9 8:96.9 9:96.8

Training KNN(skew correction & Hog) model using 1500 training data, which is 30.00% ...
3500: k:Acc => 1:97.0 2:96.9 3:97.2 4:97.4 5:97.3 6:97.4 7:97.3 8:97.3 9:97.1

Training KNN(skew correction & Hog) model using 2000 training data, which is 40.00% ...
3000: k:Acc => 1:97.4 2:97.0 3:97.5 4:97.6 5:97.7 6:97.8 7:97.8 8:97.6 9:97.5

Training KNN(skew correction & Hog) model using 2500 training data, which is 50.00% ...
2500: k:Acc => 1:97.5 2:97.3 3:97.8 4:97.8 5:97.7 6:97.8 7:97.9 8:97.7 9:97.6

Training KNN(skew correction & Hog) model using 3000 training data, which is 60.00% ...
2000: k:Acc => 1:97.7 2:97.5 3:97.8 4:97.8 5:97.8 6:97.7 7:98.0 8:97.9 9:97.8

Training KNN(skew correction & Hog) model using 3500 training data, which is 70.00% ...
1500: k:Acc => 1:97.6 2:97.5 3:97.9 4:97.8 5:97.9 6:97.8 7:98.0 8:97.9 9:98.0

Training KNN(skew correction & Hog) model using 4000 training data, which is 80.00% ...
1000: k:Acc => 1:97.3 2:97.6 3:97.5 4:97.4 5:97.4 6:97.5 7:97.5 8:97.5 9:97.7

Training KNN(skew correction & Hog) model using 4500 training data, which is 90.00% ...
500: k:Acc => 1:97.6 2:98.6 3:98.0 4:98.2 5:97.6 6:98.0 7:97.6 8:97.8 9:98.0

결과(1): 2_4_knn 예제에서 사용한 deskew와 같은 크기(144)의 hog 기술자를 사용했을 경우
90% 학습시켰을 때: SVM=99.00%, 2-NN=98.6(best), 1-NN:97.6(worst)
80% 학습시켰을 때: SVM=98.70%, 9-NN=97.7(best), 1-NN:97.3(worst)
70% 학습시켰을 때: SVM=98.87%, 7-NN=98.0(best), 1-NN:97.6(worst)
60% 학습시켰을 때: SVM=98.70%, 7-NN=98.0(best), 2-NN:97.5(worst)
50% 학습시켰을 때: SVM=98.60%, 7-NN=97.9(best), 2-NN:97.3(worst)
40% 학습시켰을 때: SVM=98.50%, 6-NN=97.8(best), 2-NN:97.0(worst)
30% 학습시켰을 때: SVM=98.00%, 4-NN=97.4(best), 1-NN:97.0(worst)
20% 학습시켰을 때: SVM=98.78%, 4-NN=97.4(best), 1-NN:96.7(worst)
10% 학습시켰을 때: SVM=98.13%, 4-NN=96.1(best), 8-NN:95.7(worst)
==> 전반적으로 KNN에서 K를 잘 선택했을 때 보다도 우수하다.





결과(2): hog의 크기를 더 크게하면 미미하지만 좀더 정확도가 정확도가 증가하는 모습을 관찰할 수 있다.
    50%를 학습에 사용했을 때 정확도 98.60%, descriptor size: 144:
        hog = cv2.HOGDescriptor((SIZE_IMAGE, SIZE_IMAGE), (8, 8), (4, 4), (8, 8), 9, 1, -1, 0, 0.2, 1, 64, True)
    90%를 학습에 사용했을 때 정확도 99.00%, descriptor size: 144:
        hog = cv2.HOGDescriptor((SIZE_IMAGE, SIZE_IMAGE), (8, 8), (4, 4), (8, 8), 9, 1, -1, 0, 0.2, 1, 64, True)

    50%를 학습에 사용했을 때 정확도 98.71 %, descriptor size: 719:
        hog = cv2.HOGDescriptor((SIZE_IMAGE, SIZE_IMAGE), (12, 12), (4, 4), (4, 4), 9, 1, -1, 0, 0.2, 1, 64, True)
    90%를 학습에 사용했을 때 정확도 99.4 %, descriptor size: 719:
        hog = cv2.HOGDescriptor((SIZE_IMAGE, SIZE_IMAGE), (12, 12), (4, 4), (4, 4), 9, 1, -1, 0, 0.2, 1, 64, True)



참고: 이 예제와 거의 같은 예제가 다음 링크에 소개 되어 있다.
    Handwritten Digits Classification : An OpenCV ( C++ / Python ) Tutorial
        https://learnopencv.com/handwritten-digits-classification-an-opencv-c-python-tutorial/


HoG 원리 설명 자료
    Histogram of Oriented Gradients explained using OpenCV
        https://learnopencv.com/histogram-of-oriented-gradients/


HOGDescriptor()
    creates the HOG descriptor and detector with default params.
    default: HOGDescriptor(Size(64,128), Size(16,16), Size(8,8), Size(8,8), 9 )

HOGDescriptor() [2/4]
    cv::HOGDescriptor::HOGDescriptor(Size 	_winSize,
    Size 	_blockSize,
    Size 	_blockStride,
    Size 	_cellSize,
    int 	_nbins,
    int 	_derivAperture = 1,
    double 	_winSigma = -1,
    HOGDescriptor::HistogramNormType 	_histogramNormType = HOGDescriptor::L2Hys,
    double 	_L2HysThreshold = 0.2,
    bool 	_gammaCorrection = false,
    int 	_nlevels = HOGDescriptor::DEFAULT_NLEVELS,
    bool 	_signedGradient = false
    )


Computes HOG descriptors of given image.
virtual void cv::HOGDescriptor::compute	(InputArray 	img,
    std::vector< float > & 	descriptors,
    Size 	winStride = Size(),
    Size 	padding = Size(),
    const std::vector< Point > & 	locations = std::vector< Point >()
    )
    Parameters
        img:	Matrix of the type CV_8U containing an image where HOG features will be calculated.
        descriptors:	Matrix of the type CV_32F
        winStride:	Window stride. It must be a multiple of block stride.
        padding:	Padding
        locations:	Vector of Point


detectMultiScale() [1/2]
참고: HOG detectMultiScale parameters explained
    https://www.pyimagesearch.com/2015/11/16/hog-detectmultiscale-parameters-explained/
virtual void cv::HOGDescriptor::detectMultiScale(InputArray 	img,
    std::vector< Rect > & 	foundLocations,
    std::vector< double > & 	foundWeights,
    double 	hitThreshold = 0,
    Size 	winStride = Size(),
    Size 	padding = Size(),
    double 	scale = 1.05,
    double 	finalThreshold = 2.0,
    bool 	useMeanshiftGrouping = false
    )

Parameters
    img:	Matrix of the type CV_8U or CV_8UC3 containing an image where objects are detected.
    foundLocations:	Vector of rectangles where each rectangle contains the detected object.
    foundWeights:	Vector that will contain confidence values for each detected object.
    hitThreshold:	Threshold for the distance between features and SVM classifying plane. Usually it is 0 and should be specified in the detector coefficients (as the last free coefficient). But if the free coefficient is omitted (which is allowed), you can specify it manually here.
    winStride:	Window stride. It must be a multiple of block stride.
    padding:	Padding
    scale:	Coefficient of the detection window increase.
    finalThreshold:	Final threshold
    useMeanshiftGrouping:	indicates grouping algorithm

• img: 입력 영상. cv2.CV_8UC1 또는 cv2.CV_8UC3.
• hitThreshold: 특징 벡터와 SVM 분류 평면까지의 거리에 대한 임계값
• winStride: 셀 윈도우 이동 크기. (0, 0) 지정 시 셀 크기와 같게 설정.
• padding: 패딩 크기
• scale: 검색 윈도우 크기 확대 비율. 기본값은 1.05.
• finalThreshold: 검출 결정을 위한 임계값
• useMeanshiftGrouping: 겹쳐진 검색 윈도우를 합치는 방법 지정 플래그
• foundLocations: (출력) 검출된 사각형 영역 정보
• foundWeights: (출력) 검출된 사각형 영역에 대한 신뢰도


"""

# Import required packages:
import cv2
import numpy as np

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
    """Trains the model using the samples and the responses"""

    model.train(samples, cv2.ml.ROW_SAMPLE, responses)
    return model


def svm_predict(model, samples):
    """Predicts the response based on the trained model"""

    return model.predict(samples)[1].ravel()


def svm_evaluate(model, samples, labels):
    """Calculates the accuracy of the trained model"""

    predictions = svm_predict(model, samples)
    accuracy = (labels == predictions).mean()
    print('Percentage Accuracy: %.2f %%' % (accuracy * 100))

def get_hog():
    # Get hog descriptor
    # cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins, derivAperture, winSigma, histogramNormType,
    # L2HysThreshold, gammaCorrection, nlevels, signedGradient)

    # descriptor size: 144, Percentage Accuracy: 98.60 %
    hog = cv2.HOGDescriptor((SIZE_IMAGE, SIZE_IMAGE), (8, 8), (4, 4), (8, 8), 9, 1, -1, 0, 0.2, 1, 64, True)

    # descriptor size: 144, Percentage Accuracy: 97.96 %
    #hog = cv2.HOGDescriptor((SIZE_IMAGE, SIZE_IMAGE), (16, 16), (4, 4), (8, 8), 9, 1, -1, 0, 0.2, 1, 64, True)

    # get descriptor size: 441, Percentage Accuracy: 95.56 %
    #hog = cv2.HOGDescriptor((SIZE_IMAGE, SIZE_IMAGE), (8, 8), (2, 2), (8, 8), 9, 1, -1, 0, 0.2, 1, 64, True)

    # get descriptor size: 324, Percentage Accuracy: 97.92 %
    #hog = cv2.HOGDescriptor((SIZE_IMAGE, SIZE_IMAGE), (10, 10), (2, 2), (10, 10), 9, 1, -1, 0, 0.2, 1, 64, True)

    # get descriptor size: 576, Percentage Accuracy: 92.88 %
    #hog = cv2.HOGDescriptor((SIZE_IMAGE, SIZE_IMAGE), (6, 6), (2, 2), (6, 6), 9, 1, -1, 0, 0.2, 1, 64, True)

    # get descriptor size: 900, Percentage Accuracy: 97.80 %
    #hog = cv2.HOGDescriptor((SIZE_IMAGE, SIZE_IMAGE), (12, 12), (2, 2), (6, 6), 9, 1, -1, 0, 0.2, 1, 64, True)

    # get descriptor size: 324, Percentage Accuracy: 98.60 %
    #hog = cv2.HOGDescriptor((SIZE_IMAGE, SIZE_IMAGE), (18, 18), (2, 2), (6, 6), 9, 1, -1, 0, 0.2, 1, 64, True)

    # get descriptor size: 36, Percentage Accuracy: 53.32 %
    #hog = cv2.HOGDescriptor((SIZE_IMAGE, SIZE_IMAGE), (19, 19), (1, 1), (19, 19), 9, 1, -1, 0, 0.2, 1, 64, True)

    # get descriptor size: 225, Percentage Accuracy: 94.76 %
    #hog = cv2.HOGDescriptor((SIZE_IMAGE, SIZE_IMAGE), (4, 4), (4, 4), (4, 4), 9, 1, -1, 0, 0.2, 1, 64, True)

    # get descriptor size: 576, Percentage Accuracy: 97.76 %
    #hog = cv2.HOGDescriptor((SIZE_IMAGE, SIZE_IMAGE), (8, 8), (4, 4), (4, 4), 9, 1, -1, 0, 0.2, 1, 64, True)

    # get descriptor size: 729, Percentage Accuracy: 98.72 %  ---------  best!!
    #hog = cv2.HOGDescriptor((SIZE_IMAGE, SIZE_IMAGE), (12, 12), (4, 4), (4, 4), 9, 1, -1, 0, 0.2, 1, 64, True)

    #hog = cv2.HOGDescriptor((SIZE_IMAGE, SIZE_IMAGE), (16, 16), (4, 4), (8, 8), 9, 1, -1, 0, 0.2, 1, 64, True)

    #"""
    # 2_4_KNN 예제에서 사용하였던 hog 설정값과 같다.
    hog = cv2.HOGDescriptor(
            _winSize=(SIZE_IMAGE, SIZE_IMAGE),   # 디스크립터를 만들고자 하는 영상의 크기, (20, 20).
            _blockSize=(8, 8),       # h x w pixels. 정규화가 일어나는 단위, 각각의 cell은 여러 개의 block에 중첩되어 반영될 수 있다.
            _blockStride=(4, 4),
            _cellSize=(8, 8),        # h x w pixels. 1-D 히스토그램을 산출하는 기본 단위
            _nbins=9,                # number of orientation bins
            _derivAperture=1,        # default
            _winSigma=-1,            # default
            _histogramNormType=0,    # default
            _L2HysThreshold=0.2,     # default
            _gammaCorrection=1,      # default
            _nlevels=64,             # default
            _signedGradient=True)
    #"""

    print(f"hog descriptor size={hog.getDescriptorSize()}")
    return hog


def raw_pixels(img):
    """Return raw pixels as feature from the image"""

    return img.flatten()


if __name__ == '__main__':
    # 1) Load all the digits and the corresponding labels:
    digits, labels = load_digits_and_labels('digits.png')

    # 2) Shuffle data
    # Constructs a random number generator:
    rand = np.random.RandomState(1234)
    # Randomly permute the sequence:
    shuffle = rand.permutation(len(digits))
    digits, labels = digits[shuffle], labels[shuffle]

    # 3) Create a HoG feature descriptor:
    hog = get_hog()
    #hog = get_hog2()

    # 4) Compute the descriptors for all the images.
    # In this case, the HoG descriptor is calculated
    hog_descriptors = []
    for img in digits:
        hog_descriptors.append(hog.compute(deskew(img)))
    #print(1, len(hog_descriptors), type(hog_descriptors[0]), hog_descriptors[0].shape); exit()
    #hog_descriptors = np.squeeze(hog_descriptors)      # squeeze() 할 필요없어 보여 삭제함.
    #print(2, len(hog_descriptors))

    # 5) 데이터를 학습용과 테스트 용으로 나눈다. 이들은 각각 영상과 레이블을 가진다.
    # At this point we split the data into training and testing:
    partition = int(0.9 * len(hog_descriptors))     # 90%를 학습, 10%를 테스트로 사용한다. 99.00%
    #partition = int(0.8 * len(hog_descriptors))     # 80%를 학습, 20%를 테스트로 사용한다. 98.70%
    #partition = int(0.7 * len(hog_descriptors))     # 70%를 학습, 30%를 테스트로 사용한다. 98.87%
    #partition = int(0.6 * len(hog_descriptors))     # 60%를 학습, 40%를 테스트로 사용한다. 98.70%
    #partition = int(0.5 * len(hog_descriptors))     # 50%를 학습, 50%를 테스트로 사용한다. 98.60%
    #partition = int(0.4 * len(hog_descriptors))     # 40%를 학습, 60%를 테스트로 사용한다. 98.50%
    #partition = int(0.3 * len(hog_descriptors))    # 30%를 학습, 70%를 테스트로 사용한다. 98.00%
    #partition = int(0.2 * len(hog_descriptors))     # 20%를 학습, 80%를 테스트로 사용한다. 97.78%
    #partition = int(0.1 * len(hog_descriptors))     # 10%를 학습, 90%를 테스트로 사용한다. 96.13%

    # np.split(in_array, [x1, x2]): in_array를 [:x1], [x1:x2], [x3:] 총 3개의 어레이로 나누어 반환한다.
    hog_descriptors_train, hog_descriptors_test = np.split(hog_descriptors, [partition])
    labels_train, labels_test = np.split(labels, [partition])
    print("5.1) 학습데이터:", type(hog_descriptors_train), hog_descriptors_train.shape, hog_descriptors_train.dtype)
    # (4500, 144) float32
    print("5.2) 학습레이블:", type(labels_train), labels_train.shape)    #  (4500,)


    # 6) SVM 모델 객체를 하나 생성한다.
    print('Training SVM model ...')
    model = svm_init(C=12.5, gamma=0.50625)

    # 7) 학습 데이터로 모델을 학습한다.
    #svm_train(model, hog_descriptors_train, labels_train)
    model.train(hog_descriptors_train, cv2.ml.ROW_SAMPLE, labels_train)


    print('Evaluating model ... ')
    #svm_evaluate(model, hog_descriptors_test, labels_test)
    predictions = model.predict(hog_descriptors_test)[1].ravel()
    # predict()가 반환하는 것은 2개의 원로로 이루어진 튜플 정보이다. (retval, result)
    # 그중 results.shape=(test_data_길이, 1)
    accuracy = (labels_test == predictions).mean()
    print('Percentage Accuracy: %.2f %%' % (accuracy * 100))

