from keras import Model, Sequential
from keras.applications import MobileNetV2
from keras.layers import Dense, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, BatchNormalization

from config import HEIGHT, WIDTH


def image_net_model():
    input_shape = (HEIGHT, WIDTH, 1)
    vgg16 = MobileNetV2(weights=None, include_top=True, input_shape=input_shape)
    # Add a layer where input is the output of the  second last layer
    x = Dense(3, activation='softmax', name='predictions')(vgg16.layers[-2].output)

    # Then create the corresponding model
    my_model = Model(input=vgg16.input, output=x)
    my_model.summary()
    my_model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])
    return my_model

def conv2d_model(output_len = 3):
    input_shape = (HEIGHT, WIDTH, 1)

    model = Sequential()
    model.add(Conv2D(128, kernel_size=(3, 3),
                     activation='relu',
                     input_shape=input_shape))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))


    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(output_len, activation='softmax'))

    model.compile(loss="categorical_crossentropy",
                  optimizer="adam",
                  metrics=['accuracy'])
    return model