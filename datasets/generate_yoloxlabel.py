# from utils import *
import os
from pathlib import Path
import json
import numpy as np
import scipy
import cv2 as cv
from matplotlib import pyplot as plt
import matplotlib.patches as patches
from torchvision import transforms

# from datasets.speed import SpeedDataset
from spark import SparkDataset
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from tqdm import tqdm

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def pinhole_model(K, t):
    p = K @ t
    return p / p[-1]


def generate_label(K, pose, list_file):
    vertices_b = np.array([[-0.79, -0.3, -0.38],
                           [-0.79, -0.3, 0.4],
                           [-0.3, 0.3, -0.38],
                           [-0.3, 0.3, 0.4],
                           [0.79, -0.3, -0.38],
                           [0.79, -0.3, 0.4],
                           [0.35, 0.3, -0.38],
                           [0.35, 0.3, 0.4]]) * 0.4
    q = Quaternion(pose[3:])
    t = pose[0:3]

    vertices_c = q.rotation_matrix @ vertices_b.transpose() + t[np.newaxis].repeat(8, axis=0).transpose()
    vertices_uv = pinhole_model(K, vertices_c).transpose()

    x1, y1 = tuple(vertices_uv[:, :-1].min(axis=0))
    x2, y2 = tuple(vertices_uv[:, :-1].max(axis=0))

    b = (x1, y1, x2, y2)
    cls_id = 0
    list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str(cls_id))
    list_file.write(" " + ",".join([str(a) for a in pose]))
    list_file.write('\n')


if __name__ == "__main__":

    label_txt = "./datasets/spark_2022_stream_2/train.txt"
    list_file = open(label_txt, 'w', encoding='utf-8')
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])

    imgs_dir = "./datasets/spark_2022_stream_2/train/images"
    labels_file = "./datasets/spark_2022_stream_2/train/train.csv"
    train_ds = SparkDataset(imgs_dir, labels_file, augment=False, data_transform=transform)
    K = train_ds.K

    for k in tqdm(range(train_ds.__len__())):
        meta_info = train_ds.getitem1(k)
        img_path = os.path.join(imgs_dir, meta_info["seq_name"], meta_info["img_name"])
        pose = meta_info["pose"]

        list_file.write(img_path)
        generate_label(K, pose, list_file)
    list_file.close()