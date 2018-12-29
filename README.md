<h4 id="0">0. 背景</h4>

之前写过一个 keras 进行图像分类的教程，同时也便于自己使用，进行了开源。经过一段时间的学习，虽然已不再使用 keras 和 tensorflow 作为深度学习框架进行项目开发，但是 keras 的简洁性还是值得新手选择使用的。这里完善一下教程和代码。**建议：框架只是工具而已，还是多学理论和论文的好**

完整代码：[keras_image_classifier](https://github.com/spytensor/keras_image_classifier)

完整教程：[利用 keras 从实例掌握深度学习图像分类](http://www.spytensor.com/index.php/archives/30/)

个人博客：[超杰](http://www.spytensor.com/)

<h4 id="1">1. 更新</h4>

2018年12月29日 第一次更新：更新全部文档和代码

<h4 id="2">2. 声明</h4>

开源只是帮助一些需要帮助的人，如果有疑问欢迎咨询，但是代码已做过线下调试，确认无误才进行发布的，在使用过程中如果遇到一些 bug 请自行谷歌或百度，如果仍有疑问欢迎联系我，联系方式：zhuchaojie@buaa.edu.cn

另外深度学习也好机器学习也好，请学点编程吧。。。。

<h4 id="3">3. 数据格式</h4>

为了便于大家使用相同的数据进行训练，从而熟悉整个过程，然后再转而到自己的实际项目上，这里了提供公开数据集`交通标志数据集` ，下载链接 [traffic-sign](https://pan.baidu.com/s/1Qe5THKTmDOyonzIkPVe0Og)
数据存储格式：


- data/
    - train/
        - 00000/
        - 00001/
        - 00002/
        - ...
    - test/
        - 00000/
        - 00001/
        - 00002/
        

关于数据集的划分：为了验证模型的效果，我们需要再另设验证集，以便在训练过程中验证模型效果，但是最终的评测结果，需要在测试集上进行，所以我们使用 `sklearn.model_selection.train_test_split()` 函数对训练集进行划分，比例采用 `训练集：验证集 = 7:3`

<h4 id="4">4. 项目结构介绍</h4>

- checkpoints/
- config.py
- model.py
- data.py
- main.py

<h5 id="4.1">4.1 checkpoints</h5>

主要用来存放训练好的模型权重，默认只保存模型权重不保存整个网络结构。

<h5 id="4.2">4.2 config.py</h5>

超参数文件，本文件定义了整个项目要用到的超参数，具体如下:

```python
### define global configs  ###

class DefaultConfigs(object):
    #1. string configs
    train_data = "../data/train/"
    val_data = "../data/val/"  # if exists else use train_test_split to generate val dataset
    test_data = "../data/all/traffic-sign/test/00003/"  # for competitions
    model_name = "NASNetMobile"
    weights_path = "./checkpoints/model.h5"#save weights for predict
    
    #2. numerical configs
    lr = 0.001
    epochs = 50
    num_classes = 62
    image_size = 224
    batch_size = 16
    channels = 3
    gpu = "0"
    
config = DefaultConfigs()
```

<h5 id="4.3">4.3 model.py</h5>

**模型搭建文件** 。由于日常任务中常使用预训练好的模型并进行 finetune ，这里只提供使用该版本，至于自己搭建模型，不是本教程的重点所在，如果有需要的请自行谷歌。

提醒：这里为了方便进行更复杂的分类任务，提供了一个 ensemble 版本的模型搭建过程。

```python

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
    out1 = GlobalMaxPooling2D()(x)  # GMP feature
    out2 = GlobalAveragePooling2D()(x) # GAP feature
    out3 = Flatten()(x)                # Flatten feature
    out = Concatenate(axis=-1)([out1, out2, out3])  #concate all feature
    out = Dropout(0.5)(out)
    out = Dense(config.num_classes, activation="softmax", name="classifier")(out)
    model = Model(inputs, out)
    model.compile(optimizer=Adam(0.0001), loss=categorical_crossentropy, metrics=['acc'])

    return model
```
<h5 id="4.4">4.4 data.py</h5>

自定义的`data generator`模块。在之前，个人喜欢将所有数据直接加载到内存中，然后再进训练，这样做的好处就是训练过程减弱了大量数据的频繁调度问题，但是如果数据量过大，内存吃不消就行不通了，而且在调试过程中也很麻烦，这里使用 python 的 `yield` 机制，能够避免小机无法运行的问题。

本模块共包含三个部分：`get_files()` `augument()` `create_train()`

1. `get_files()` 详情如下：

主要功能是循环读取每个文件夹下的图片，并根据路径信息提取类别信息。

**提醒**：windows系统请自行修改类别获取方法，如果不会，请百度或谷歌

修改位置 `labels.append(int(file.split("/")[-2]))`

```
def get_files(root,mode):
    #for test
    if mode == "test":
        files = []
        for img in os.listdir(root):
            files.append(root + img)
        files = pd.DataFrame({"filename":files})
        return files
    elif mode != "test":
        #for train and val       
        all_data_path,labels = [],[]
        image_folders = list(map(lambda x:root+x,os.listdir(root)))
        all_images = list(chain.from_iterable(list(map(lambda x:glob(x+"/*"),image_folders))))
        print("loading train dataset")
        for file in tqdm(all_images):
            all_data_path.append(file)
            labels.append(int(file.split("/")[-2]))
        all_files = pd.DataFrame({"filename":all_data_path,"label":labels})
        return all_files
    else:
        print("check the mode please!")

```

2. `augument()`

使用的线上数据增强方式，见开源 [imgaug](https://github.com/aleju/imgaug),此处不做重复介绍。

3. `create_train()`

使用python的yield机制对数据进行加载，函数定义如下：

```python

class data_generator:
    def __init__(self,data_lists,mode,augument=True):
        self.mode = mode
        self.augment = augument
        self.all_files = data_lists
    def create_train(self):
        images = []
        dataset_info = self.all_files.values
        #embed()
        """
        if not self.mode == "test":
            for index,row in all_files.iterrows():
                images.append((row["filename"],row["label"]))
        else:
            for index,row in all_files.iterrows():
                images.append((row["filename"]))
        """
        while 1:
            shuffle(dataset_info)
            #print(dataset_info)
            for start in range(0,len(dataset_info),config.batch_size):
                end = min(start + config.batch_size,len(dataset_info))
                batch_images = []
                X_train_batch = dataset_info[start:end]
                batch_labels = np.zeros((len(X_train_batch),config.num_classes))
                for i in range(len(X_train_batch)):
                    #print(X_train_batch[i])
                    image = cv2.imread(X_train_batch[i][0])
                    image = cv2.resize(image,(config.image_size,config.image_size),interpolation=cv2.INTER_NEAREST)

                    if self.augument:
                        image = self.augument(image)
                    batch_images.append(image / 255.)
                    if not self.mode == "test":
                        batch_labels[i][X_train_batch[i][1]] = 1
                        #print(np.array(batch_images).shape)
                yield np.array(batch_images, np.float32), batch_labels

```

<h5 id="4.5">4.5 main.py</h5>

包含功能：

- create_callbacks(),主要实现`提前停止训练`、`权重保存`、`学习率衰减`
- train(),训练函数 ，执行训练
- test()，测试函数，返回每个图片的预测类别

```
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

```

<h4 id="5">5 使用方法</h4>

1. 下载数据集并解压 [traffic-sign](https://pan.baidu.com/s/1Qe5THKTmDOyonzIkPVe0Og)
2. 修改 `config.py` 中 `train_data`，`test_data` 路径，其中 test_data 示例：`test_data = "../data/all/traffic-sign/test/00003/"`
3. 训练：`python main.py`
4. 预测：修改 `main.py` 中 76 行的 `mode = "test"` 并执行 `python main.py`


