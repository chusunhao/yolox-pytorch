import os
import torch
import skimage.io
from skimage.io import imread
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from imgaug import augmenters as iaa
from pyquaternion import Quaternion
from augment import rotate_cam


class SparkDataset(Dataset):
    def __init__(self, imgs_dir, labels_file, augment=True, data_transform=None, seq=None):
        super(SparkDataset, self).__init__()
        self.imgs_dir = imgs_dir
        self.labels_file = labels_file
        df = pd.read_csv(labels_file)
        if seq is None:
            self.df = df
        else:
            self.df = df[df["sequence"] == seq]

        self.augment = augment
        self.transform = data_transform
        self.seq = seq
        self.K = np.array([[1744.92206139719, 0, 737.272795902663], [0, 1746.58640701753, 528.471960188736], [0, 0, 1]])

        self.pos = np.array([self.df["Tx"], self.df["Ty"], self.df["Tz"]]).transpose()
        self.quat = np.array([self.df["Qw"], self.df["Qx"], self.df["Qy"], self.df["Qz"]]).transpose()
        self.img_name = np.array(self.df["filename"])
        self.seq_name = np.array(self.df["sequence"])

    def getitem1(self, idx):
        img_name = self.img_name[idx]
        seq_name = self.seq_name[idx]
        img_path = os.path.join(self.imgs_dir, seq_name, img_name)
        img = skimage.io.imread(img_path)
        t = self.pos[idx]
        q = self.quat[idx]

        if self.augment and np.random.rand(1) > 0.5:
            # Camera Jitter Augmentation
            axis = np.random.rand(3)
            angle = 30 * (np.random.rand(1) - 0.5)
            R_change = Quaternion(axis=axis, degrees=angle).rotation_matrix
            img, t, q = rotate_cam(img, t, q, self.K, R_change)

        if self.augment and np.random.rand(1) > 0.5:
            # Image Augmentation Pipeline
            aug_pipeline = iaa.Sequential([
                iaa.AdditiveGaussianNoise(scale=0.1 * 255),
                iaa.GaussianBlur(sigma=(0.0, 1.5)),
                iaa.Add((-40, 40)),
                iaa.Multiply((0.5, 2.0)),
                iaa.CoarseDropout([0.0, 0.03], size_percent=(0.02, 0.1)),
                iaa.AddElementwise((-40, 40)),  # to the pixels
                # iaa.ElasticTransformation(alpha=90, sigma=9),  # water-like effect
                # iaa.Cutout(),  # replace one squared area within the image by a constant intensity value
                # iaa.Dropout(p=(0, 0.2)),
                # iaa.Salt(0.1),
                # iaa.Pepper(0.1)
            ], random_order=True).to_deterministic()
            img = aug_pipeline.augment_image(img)

        img = self.transform(img)
        pose = np.concatenate([t, q], dtype=np.float32)
        return {"img_name": img_name,
                "seq_name": seq_name,
                "img": img,
                "pose": pose}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return self.getitem1(idx)
