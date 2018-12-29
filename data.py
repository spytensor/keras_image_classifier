import os
import cv2
import keras
import pandas as pd 
import numpy as np
import imgaug as ia
from config import config
from numpy.random import shuffle
from imgaug import augmenters as iaa
from itertools import chain 
from tqdm import tqdm 
from glob import glob
from IPython import embed
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

    def augument(self,image):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug)
        augment_img = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5), # horizontally flip 50% of all images
                iaa.Flipud(0.2), # vertically flip 20% of all images
                sometimes(iaa.Affine(
                    scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)}, # translate by -20 to +20 percent (per axis)
                    rotate=(-10, 10), # rotate by -45 to +45 degrees
                    shear=(-5, 5), # shear by -16 to +16 degrees
                    order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                    [
                        sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 1.0)), # blur images with a sigma between 0 and 3.0
                            iaa.AverageBlur(k=(3, 5)), # blur image using local means with kernel sizes between 2 and 7
                            iaa.MedianBlur(k=(3, 5)), # blur image using local medians with kernel sizes between 2 and 7
                        ]),
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.9, 1.1)), # sharpen images
                        iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                        # search either for all edges or for directed edges,
                        # blend the result with the original image using a blobby mask
                        iaa.SimplexNoiseAlpha(iaa.OneOf([
                           iaa.EdgeDetect(alpha=(0.5, 1.0)),
                           iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                        ])),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.01*255), per_channel=0.5), # add gaussian noise to images
                            iaa.OneOf([
                            iaa.Dropout((0.01, 0.05), per_channel=0.5), # randomly remove up to 10% of the pixels
                            iaa.CoarseDropout((0.01, 0.03), size_percent=(0.01, 0.02), per_channel=0.2),
                        ]),
                        iaa.Invert(0.01, per_channel=True), # invert color channels
                        iaa.Add((-2, 2), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                        iaa.AddToHueAndSaturation((-1, 1)), # change hue and saturation
                        # either change the brightness of the whole image (sometimes
                        # per channel) or change the brightness of subareas
                        iaa.OneOf([
                            iaa.Multiply((0.9, 1.1), per_channel=0.5),
                            iaa.FrequencyNoiseAlpha(
                                exponent=(-1, 0),
                                first=iaa.Multiply((0.9, 1.1), per_channel=True),
                                second=iaa.ContrastNormalization((0.9, 1.1))
                            )
                        ]),
                        sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                        sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                        sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                    ],
                    random_order=True
                 )
            ],
            random_order=True
        )
        image_aug = augment_img.augment_image(image)
        return image_aug

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
