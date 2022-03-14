from keras.layers import Conv2D,BatchNormalization,Activation
from keras.layers import UpSampling2D,MaxPool2D,AveragePooling2D,GlobalAveragePooling2D,MaxPooling2D
from keras.layers import Add,Multiply,Lambda
from keras.layers import Input,Dense,Flatten,Dropout
from keras.models import Model
from keras.regularizers import l2

def residual_block(input, input_channels=None, output_channels=None, kernel_size=(3, 3), stride=1):
    if output_channels is None:
        output_channels = input.shape[-1]
    if input_channels is None:
        input_channels = output_channels // 4

    strides = (stride, stride)

    x = BatchNormalization()(input)
    x = Activation('relu')(x)
    x = Conv2D(input_channels, (1, 1))(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(input_channels, kernel_size, padding='same', strides=stride)(x)

    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv2D(output_channels, (1, 1), padding='same')(x)

    if input_channels != output_channels or stride != 1:
        input = Conv2D(output_channels, (1, 1), padding='same', strides=strides)(input)

    x = Add()([x, input])
    return x

	

input_ = Input(shape=(200,200,3))
x = Conv2D(64,(3,3),input_shape=(200,200,3),activation='relu',padding='same')(input_)
x= Conv2D(64,(3,3),input_shape=(200,200,3),activation='relu',padding='same')(x)
x = BatchNormalization()(x)
x = MaxPool2D(pool_size=(3,3),strides=2)(x)
x=MaxPooling2D()(x)

x=Conv2D(128,(3,3),activation='relu',padding='same')(x)
x=Conv2D(128,(3,3),activation='relu')(x)
x=Conv2D(128,(3,3),activation='relu')(x)
x=BatchNormalization()(x)
x=MaxPooling2D(3)(x)
x=MaxPooling2D()(x)

x=Conv2D(256,(3,3),activation='relu',padding='same')(x)
x=Conv2D(256,(3,3),activation='relu')(x)
x=Conv2D(256,(3,3),activation='relu')(x)
x=BatchNormalization()(x)
x = AveragePooling2D(pool_size=(3,3), strides=(1, 1))(x)  # 1x1
x=GlobalAveragePooling2D()(x)

x = residual_block(x, input_channels=32, output_channels=64)
# x = attention_block(x, encoder_depth=1)

x=GlobalAveragePooling2D()(x)
x = Flatten()(x)
x=Dense(64,activation='relu')(x)
x=Dropout(0.4)(x)
output = Dense(2, activation='softmax')(x)

model = Model(input_, output)
