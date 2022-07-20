import numpy as np
import random
import cv2
import matplotlib.pyplot as plt
from pyquaternion import Quaternion

from torchvision import transforms
# import albumentations as A
# import albumentations as A
# from albumentations.augmentations.geometric.transforms import Affine
# from albumentations.pytorch import ToTensorV2
from imgaug import augmenters as iaa


# Augmentations
def rotate_cam(image, t, q, K, R_change):
    """ Apply warping corresponding to a random camera rotation
    Arguments:
     - image: Input image
     - t, q: Object pose (location,orientation)
     - K: Camera intrinsics matrix
     - R_change: camera rotation matrix
    Return:
        - image_warped: Output image
        - t_new, q_new: Updated object pose
    """

    # Construct perspective matrix
    h = K @ R_change.T @ np.linalg.inv(K)

    height, width = image.shape[:2]
    # Update pose
    t_new = R_change.T @ t
    puv = pinhole_model(K, t_new)
    if not ((0 < puv[0] < width) or (0 < puv[1] < height)):
        return image, t, q
    else:
        image_warped = cv2.warpPerspective(image, h, (width, height))

        q_change = Quaternion(matrix=R_change)
        q_new = (q_change.conjugate * Quaternion(q)).q

        return image_warped, t_new, q_new

def pinhole_model(K, t):
    p = K @ t
    return p/p[-1]
