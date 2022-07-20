# from utils import *
import os
from pathlib import Path
import json
import numpy as np
import scipy
import cv2 as cv
from matplotlib import pyplot as plt
from torchvision import transforms

# from datasets.speed import SpeedDataset
from datasets.spark import SparkDataset
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


def pinhole_model(K, t):
    p = K @ t
    return p / p[-1]


def frame1to2():
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         std=[0.229, 0.224, 0.225])])

    train_imgs_dir = "../datasets/spark_2022_stream_2/train/images"
    train_labels_file = "../datasets/spark_2022_stream_2/train/train.csv"
    train_ds = SparkDataset(train_imgs_dir, train_labels_file, augment=False, data_transform=transform,
                            seq="GT021")

    sample1 = train_ds.getitem1(1)
    img1 = cv.imread(sample1["img_path"])
    pose1 = sample1["pose"]
    sample2 = train_ds.getitem1(2)
    img2 = cv.imread(sample2["img_path"])
    pose2 = sample2["pose"]

    # Compute relative pose from frame1 to frame2
    q1, q2 = Quaternion(pose1[3:]), Quaternion(pose2[3:])
    # q2wrt1 = (q1 * q2.conjugate).vector
    Delta_R = (q1 * q2.conjugate).rotation_matrix

    t1, t2 = pose1[0:3], pose2[0:3]
    t1_new = Delta_R.T @ t1

    K = np.array([[1744.92206139719, 0, 737.272795902663], [0, 1746.58640701753, 528.471960188736], [0, 0, 1]])

    puv1 = pinhole_model(K, t1)
    puv1_new = pinhole_model(K, t1_new)
    puv2 = pinhole_model(K, t2)

    h = K @ Delta_R.T @ np.linalg.inv(K)

    img1_aug = cv.warpPerspective(img1, h, (1440, 1080))

    fig, ax = plt.subplots(nrows=2, ncols=2)
    ax[0].imshow(img1), ax[0].title('img1'), ax[0].scatter(puv1[0], puv1[1])
    ax[1].subplot(222), ax[1].imshow(img1_aug), ax[1].title('img1-aug'), ax[1].scatter(puv1_new[0], puv1_new[1])
    ax[2].subplot(223), ax[2].imshow(img2), ax[2].title('img2'), plt.scatter(puv2[0], puv2[1])
    plt.show()

def visualize_axes(ax, K, q, t):
    axes_b = np.eye(3)
    axes_c = Quaternion(q).rotation_matrix @ axes_b + t[np.newaxis].repeat(3, axis=0).transpose()
    puv0 = pinhole_model(K, t)
    scale = 0.5
    colors = ["r", "g", "b"]
    axes = [r"$x$", r"$y$", r"$z$"]
    for k in range(3):
        puv = pinhole_model(K, axes_c[:, k])
        ax.arrow(puv0[0], puv0[1], scale * (puv[0] - puv0[0]),
                 scale * (puv[1] - puv0[1]), head_width=20, color=colors[k])
        ax.text((scale+0.2) * (puv[0] - puv0[0]) + puv0[0], (scale+0.2) * (puv[1] - puv0[1]) + puv0[1], axes[k],
                color=colors[k], fontsize=10)

def visualize_bboxes(ax, K, q, t):
    # vertices_b = np.array([[-0.35, -0.3, -0.38],
    #                        [-0.35, -0.3,  0.425],
    #                        [-0.35,  0.3, -0.38],
    #                        [-0.35,  0.3,  0.425],
    #                        [ 0.35, -0.3, -0.38],
    #                        [ 0.35, -0.3,  0.4],
    #                        [ 0.35,  0.3, -0.38],
    #                        [ 0.35,  0.3,  0.4]]) * 0.4

    vertices_b = np.array([[-0.79, -0.3, -0.38],
                           [-0.79, -0.3,  0.4],
                           [-0.79,  0.3, -0.38],
                           [-0.79,  0.3,  0.4],
                           [ 0.79, -0.3, -0.38],
                           [ 0.79, -0.3,  0.4],
                           [ 0.79,  0.3, -0.38],
                           [ 0.79,  0.3,  0.4]]) * 0.4

    edges = np.array([[0, 1], [2, 3], [4, 5], [6, 7],
                      [0, 4], [2, 6], [1, 5], [3, 7],
                      [0, 2], [1, 3], [4, 6], [5, 7]])
    vertices_c = Quaternion(q).rotation_matrix @ vertices_b.transpose() + t[np.newaxis].repeat(8, axis=0).transpose()
    vertices_uv = pinhole_model(K, vertices_c).transpose()

    for k in range(12):
        i, j = tuple(edges[k])
        ax.arrow(vertices_uv[i, 0], vertices_uv[i, 1],
                 vertices_uv[j, 0] - vertices_uv[i, 0], vertices_uv[j, 1] - vertices_uv[i, 1], color='y')
    for k in range(8):
        ax.scatter(vertices_uv[k, 0], vertices_uv[k, 1])
        ax.text(vertices_uv[k, 0], vertices_uv[k, 1], rf"${k}$", color="y", fontsize=10)


if __name__ == "__main__":
    transform = transforms.Compose([transforms.ToPILImage(),
                                    transforms.ToTensor(),
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225])])

    train_imgs_dir = "../datasets/spark_2022_stream_2/train/images"
    train_labels_file = "../datasets/spark_2022_stream_2/train/train.csv"
    train_ds = SparkDataset(train_imgs_dir, train_labels_file, augment=False, data_transform=transform,
                            seq="GT106")


    sample = train_ds.get(200)
    img = cv.imread(os.path.join(train_imgs_dir, sample["seq_name"], sample["img_name"]))
    pose = sample["pose"]

    # Compute relative pose from frame1 to frame2
    orig_q = Quaternion(pose[3:])

    Q = list()
    Q.append(Quaternion(axis=(1.0, 0.0, 0.0), radians=0))
    Q.append(Quaternion(axis=(1.0, 0.0, 0.0), radians=np.pi/2))
    Q.append(Quaternion(axis=(1.0, 0.0, 0.0), radians=np.pi))
    Q.append(Quaternion(axis=(1.0, 0.0, 0.0), radians=-np.pi/2))
    Q.append(Quaternion(axis=(0.0, 1.0, 0.0), radians=np.pi/2))
    Q.append(Quaternion(axis=(0.0, 0.0, 1.0), radians=np.pi/2))

    dist = []
    for q in Q:
        dist.append(Quaternion.distance(orig_q, q))

    target_q = Q[np.argmin(dist)]


    # q2wrt1 = (q1 * q2.conjugate).vector
    Delta_R = (orig_q * target_q.conjugate).rotation_matrix

    orig_t = pose[0:3]
    t_new = Delta_R.T @ orig_t

    K = np.array([[1744.92206139719, 0, 737.272795902663], [0, 1746.58640701753, 528.471960188736], [0, 0, 1]])

    puv = pinhole_model(K, orig_t)
    puv_new = pinhole_model(K, t_new)

    h = K @ Delta_R.T @ np.linalg.inv(K)

    img_aug = cv.warpPerspective(img, h, (1440, 1080))


    fig, ax = plt.subplots(nrows=1, ncols=1)
    ax.imshow(img), ax.set_title(sample["img_name"])
    visualize_axes(ax, K, orig_q, orig_t)
    visualize_bboxes(ax, K, orig_q, orig_t)


    R = Quaternion(q).rotation_matrix
    # M0 = np.array([[-1, 0, 0, 0, 0, 0],
    #                [0, 0, -1, 0, 0, 0],
    #                [0, 0, 0, 0, 0, -1]])
    M0 = np.array([[-1, 0, 0],
                   [0, -1, 0],
                   [0, 0, -1]])

    P0 = np.array([[1226, 560, 1]]).transpose()
    # M1 = np.array([[-1, 0, 0, 0, 0, 0],
    #                [0, 0, -1, 0, 0, 0],
    #                [0, 0, 0, 0, 1, 0]])
    M1 = np.array([[-1, 0, 0],
                   [0, -1, 0],
                   [0, 0, 1]])
    P1 = np.array([[1382, 569, 1]]).transpose()
    # M4 = np.array([[0, 1, 0, 0, 0, 0],
    #                [0, 0, -1, 0, 0, 0],
    #                [0, 0, 0, 0, 0, -1]])
    M4 = np.array([[1, 0, 0],
                   [0, -1, 0],
                   [0, 0, -1]])
    P4 = np.array([[1200, 264, 1]]).transpose()
    # M5 = np.array([[0, 1, 0, 0, 0, 0],
    #                [0, 0, -1, 0, 0, 0],
    #                [0, 0, 0, 0, 1, 0]])
    M5 = np.array([[1, 0, 0],
                   [0, -1, 0],
                   [0, 0, 1]])
    P5 = np.array([[1366, 259, 1]]).transpose()

    A0 = np.hstack((K @ R @ M0, -P0, np.zeros((3, 1)), np.zeros((3, 1)), np.zeros((3, 1))))
    A1 = np.hstack((K @ R @ M1, np.zeros((3, 1)), -P1, np.zeros((3, 1)), np.zeros((3, 1))))
    A4 = np.hstack((K @ R @ M4, np.zeros((3, 1)), np.zeros((3, 1)), -P4, np.zeros((3, 1))))
    A5 = np.hstack((K @ R @ M5, np.zeros((3, 1)), np.zeros((3, 1)), np.zeros((3, 1)), -P5))
    A = np.vstack((A0, A1, A4, A5))

    b = (K @ orig_t).repeat(4)[np.newaxis].transpose()
    # x1 = np.linalg.solve(np.linalg.pinv(A), b)
    # x2 = np.linalg.solve(A.T @ A, A.T @ b)
    x = np.linalg.lstsq(A, b)
    # p7 = np.array([1328, 390, 1])
    # p3 = np.array([1346, 525, 1])
    # tmp1 = orig_q.conjugate.rotation_matrix @ np.linalg.inv(K) @ p7
    # tmp2 = orig_q.conjugate.rotation_matrix @ np.linalg.inv(K) @ p3
    # tmp3 = orig_q.conjugate.rotation_matrix @ orig_t
    plt.show()

