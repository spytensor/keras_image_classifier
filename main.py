from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from keras.preprocessing.image import img_to_array
from model import get_model
from config import config
from data import data_generator,get_files
import os
import warnings
import numpy as np
from IPython import embed
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu

def create_callbacks():

    early_stop = EarlyStopping(
        monitor         =       "val_acc",
        mode            =       "auto",
        patience        =       30,
        verbose         =       1
    )

    checkpoint = ModelCheckpoint(
        filepath            =       config.weights_path,
        monitor             =       "val_acc",
        save_best_only      =       True,
        save_weights_only   =       True,
        mode                =       "max",
        verbose             =        1
    )

    lr_reducer = ReduceLROnPlateau(
        monitor         =       "val_acc",
        mode            =       "max",
        epsilon         =       0.01,
        factor          =       0.1,
        patience        =       5,
    )
    return [early_stop,checkpoint,lr_reducer]

def train(callbacks):
    #1. compile
    print("--> Compiling the model...")
    model = get_model()
    # load raw train data
    raw_train_data_lists = get_files(config.train_data,"train")
    #split raw train data to train and val
    train_data_lists,val_data_lists = train_test_split(raw_train_data_lists,test_size=0.3)
    # for train
    train_datagen = data_generator(train_data_lists,"train",augument=True).create_train()
    #embed()
    # val data
    val_datagen = data_generator(val_data_lists,"val",augument=True).create_train()  # if model can predict better on augumented data ,the model should be more reboust
    history = model.fit_generator(
        train_datagen,
        validation_data = val_datagen,
        epochs = config.epochs,
        verbose = 1,
        callbacks = callbacks,
        steps_per_epoch=len(train_data_lists) // config.batch_size,
        validation_steps=len(val_data_lists) // config.batch_size
    )
def test():
    test_data_lists = get_files(config.test_data,"test")
    test_datagen = data_generator(test_data_lists,"test",augument=False).create_train()
    model = get_model()
    model.load_weights(config.weights_path)
    predicted_labels = np.argmax(model.predict_generator(test_datagen,steps=len(test_data_lists) / 16),axis=-1)  
    print(predicted_labels) 
if __name__ == "__main__":
    if not os.path.exists("./checkpoints/"):
        os.mkdir("./checkpoints/")
    callbacks = create_callbacks()
    mode = "test"
    if mode == "train":
        train(callbacks)
    elif mode =="test":
        test()
    else:
        print("check mode!")

