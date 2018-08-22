from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras import backend as K

class LeNet:

    @staticmethod
    def build(numChannels, imgRows, imgCols,numClasses,activation="relu",weightsPath=None):

        #initialize the model
        model = Sequential()
        inputShape = (imgRows,imgCols,numChannels)

        # if we are using "channel first ", update the input shape
        if K.image_data_format() == "channels_first":
            inputShape = (numChannels, imgRows, imgCols)

        model.add(Conv2D(20,5,padding="same",input_shape=inputShape))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

        model.add(Conv2D(50, 5, padding="same", input_shape=inputShape))
        model.add(Activation(activation))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

        model.add(Flatten())
        model.add(Dense(500))
        model.add(Activation(activation))

        model.add(Dense(numClasses))

        model.add(Activation("softmax"))

        if weightsPath is not None:
            model.load_weights(weightsPath)

        return model
