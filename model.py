from keras.applications import NASNetMobile
from keras.applications import resnet50
from keras.layers import GlobalAveragePooling2D,GlobalMaxPooling2D,Dense,Flatten,Input,Concatenate,Dropout
from keras.models import Model
from keras.losses import mae, sparse_categorical_crossentropy, binary_crossentropy,categorical_crossentropy
from keras.optimizers import Adam
from config import config

def get_model():
    inputs = Input((config.image_size, config.image_size, config.channels))
    base_model = NASNetMobile(include_top=False, input_shape=(config.image_size, config.image_size, config.channels))#, weights=None
    x = base_model(inputs)
    out1 = GlobalMaxPooling2D()(x)
    out2 = GlobalAveragePooling2D()(x)
    out3 = Flatten()(x)
    out = Concatenate(axis=-1)([out1, out2, out3])
    out = Dropout(0.5)(out)
    out = Dense(config.num_classes, activation="softmax", name="classifier")(out)
    model = Model(inputs, out)
    model.compile(optimizer=Adam(0.0001), loss=categorical_crossentropy, metrics=['acc'])

    return model
