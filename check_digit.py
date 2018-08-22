from lenet import LeNet
from sklearn.model_selection import train_test_split
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.utils import np_utils
from keras import backend as K
import numpy as np
import argparse
import cv2


drawing=False # true if mouse is pressed
mode=True # if True, draw rectangle. Press 'm' to toggle to curve

# mouse callback function
def interactive_drawing(event,x,y,flags,param):
    global ix,iy,drawing, mode

    if event==cv2.EVENT_LBUTTONDOWN:
        drawing=True
        ix,iy=x,y

    elif event==cv2.EVENT_MOUSEMOVE:
        if drawing==True:
            if mode==True:
                cv2.line(img,(ix,iy),(x,y),(255,255,255),6)
                ix, iy = x, y
    elif event==cv2.EVENT_LBUTTONUP:
        drawing=False
        if mode==True:
            cv2.line(img, (ix, iy), (x, y), (255, 255, 255), 6)
            ix, iy = x, y


ap = argparse.ArgumentParser()
ap.add_argument("-s", "--save-model", type=int, default=-1, help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1, help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str, help="(optional) path to weights file")
args = vars(ap.parse_args())

print("[INFO] downloading MNIST...")
((trainData,trainLabels),(testData,testLabels)) = mnist.load_data()

if K.image_data_format() == "chaneels_first":
    trainData = trainData.reshape((trainData.shape[0],1,28,28))
    testData = testData.reshape((testData.shape[0],1,28,28))

else:
    trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
    testData = testData.reshape((testData.shape[0], 28, 28, 1))

trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

trainLabels = np_utils.to_categorical(trainLabels, 10)
testLabels = np_utils.to_categorical(testLabels, 10)

print("[INFO] compiling model...")
opt = SGD(lr=0.01)
model = LeNet.build(numChannels=1,imgRows=28,imgCols=28,numClasses=10,
                    weightsPath=args["weights"] if args["load_model"] > 0 else None)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

if args["load_model"] < 0:
    print("[INFO] training...")
    model.fit(trainData, trainLabels, batch_size=128, epochs=20,
              verbose=1)

    # show the accuracy on the testing set
    print("[INFO] evaluating...")
    (loss, accuracy) = model.evaluate(testData, testLabels,
                                      batch_size=128, verbose=1)
    print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))

if args["save_model"] > 0:
	print("[INFO] dumping weights to file...")
	model.save_weights(args["weights"], overwrite=True)

# checking imput from drawn image
# classify the digit
while(True):

    img = np.zeros((200, 200, 3), np.uint8)
    cv2.namedWindow('Window')
    cv2.setMouseCallback('Window', interactive_drawing)
    while (1):
        cv2.imshow('Window', img)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break

    img = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY);
    # img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # thresh = 127
    # img = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
    print img.shape
    img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)

    if K.image_data_format() == "channels_first":
        img = img.reshape(1,28,28)

    # otherwise we are using "channels_last" ordering
    else:
        img = img.reshape(28, 28,1)

    probs = model.predict(img[np.newaxis, :])
    prediction = probs.argmax(axis=1)

    # merge the channels into one image
    img = cv2.merge([img] * 3)

    # resize the image from a 28 x 28 image to a 96 x 96 image so we
    # can better see it
    img = cv2.resize(img, (96, 96), interpolation=cv2.INTER_LINEAR)

    # show the image and prediction
    cv2.putText(img, str(prediction[0]), (5, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    print("[INFO] Predicted: {}".format(prediction[0]))
    cv2.imshow("Digit", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()