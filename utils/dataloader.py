from random import sample, shuffle

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from pyquaternion import Quaternion
from utils.utils import cvtColor, preprocess_input
from imgaug import augmenters as iaa


class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, epoch_length, \
                 mosaic, mixup, mosaic_prob, mixup_prob, train, special_aug_ratio=0.7):
        super(YoloDataset, self).__init__()
        self.annotation_lines = annotation_lines
        # self.check_annotation()
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.epoch_length = epoch_length
        self.mosaic = mosaic
        self.mosaic_prob = mosaic_prob
        self.mixup = mixup
        self.mixup_prob = mixup_prob
        self.train = train
        self.special_aug_ratio = special_aug_ratio

        self.epoch_now = -1
        self.length = len(self.annotation_lines)
        self.K = np.array([[1744.92206139719, 0, 737.272795902663], [0, 1746.58640701753, 528.471960188736], [0, 0, 1]])

    # def check_annotation(self):
    #     for line in self.annotation_lines:
    #         line = line.split()
    #
    #         image_path = line[0]
    #         box = np.array(list(map(int, line[1].split(','))))[np.newaxis]
    #         pose = np.array(list(map(float, line[2].split(','))))[np.newaxis]
    #         assert image_path is not None
    #         assert box.shape[-1] == 5
    #         assert pose.shape[-1] == 7

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index = index % self.length

        # ---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        # ---------------------------------------------------#
        if self.mosaic and self.rand() < self.mosaic_prob and self.epoch_now < self.epoch_length * self.special_aug_ratio:
            lines = sample(self.annotation_lines, 3)
            lines.append(self.annotation_lines[index])
            shuffle(lines)
            image, box = self.get_random_data_with_Mosaic(lines, self.input_shape)

            if self.mixup and self.rand() < self.mixup_prob:
                lines = sample(self.annotation_lines, 1)
                image_2, box_2 = self.get_random_data(lines[0], self.input_shape, random=self.train)
                image, box = self.get_random_data_with_MixUp(image, box, image_2, box_2)
        else:
            image, box, pose = self.get_random_data(self.annotation_lines[index], self.input_shape, random=self.train)

        image = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box = np.array(box, dtype=np.float32)
        if len(box) != 0:
            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        return image, box, pose

    def rand(self, a=0, b=1):
        return np.random.rand() * (b - a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line = annotation_line.split()
        # ------------------------------#
        #   读取图像并转换成RGB图像
        # ------------------------------#
        image = Image.open(line[0])
        image = cvtColor(image)
        # ------------------------------#
        #   获得图像的高宽与目标高宽
        # ------------------------------#
        iw, ih = image.size
        h, w = input_shape
        # ------------------------------#
        #   获得预测框
        # ------------------------------#
        box = np.array(list(map(int, line[1].split(','))))[np.newaxis]
        pose = np.array(list(map(float, line[2].split(','))))[np.newaxis]

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            # ---------------------------------#
            #   将图像多余的部分加上灰条
            # ---------------------------------#
            image = image.resize((nw, nh), Image.BICUBIC)
            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            # ---------------------------------#
            #   对真实框进行调整
            # ---------------------------------#
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

            return image_data, box, pose

        # ------------------------------------------#
        #   对图像进行投影变换进行增强
        # ------------------------------------------#
        warp = self.rand() < .5
        if warp:
            # Camera Jitter Augmentation
            axis = np.random.rand(3)
            angle = 30 * (np.random.rand(1) - 0.5)
            R_change = Quaternion(axis=axis, degrees=angle).rotation_matrix
            t, q = pose[0][0:3], pose[0][3:]
            image_data = np.array(image, np.uint8)
            image_data, t, q, box = self.rotate_cam(image_data, t, q, box, R_change)
            pose[0] = np.hstack((t, q))
            image = Image.fromarray(image_data)

        # ------------------------------------------#
        #   Resize和Padding
        # ------------------------------------------#
        scale = min(w / iw, h / ih)
        nw = int(iw * scale)
        nh = int(ih * scale)
        dx = (w - nw) // 2
        dy = (h - nh) // 2

        # ---------------------------------#
        #   将图像多余的部分加上灰条
        # ---------------------------------#
        image = image.resize((nw, nh), Image.BICUBIC)
        new_image = Image.new('RGB', (w, h), (128, 128, 128))
        new_image.paste(image, (dx, dy))
        image_data = np.array(new_image, np.uint8)

        # ------------------------------------------#
        #   翻转图像
        # ------------------------------------------#
        # flip = self.rand() < .5
        # if flip:
        #     image = image.transpose(Image.FLIP_LEFT_RIGHT)

        # image_data = np.array(image, np.uint8)
        # ---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        # ---------------------------------#
        use_iaa = self.rand() < 0.5
        if use_iaa:
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
            image_data = aug_pipeline.augment_image(image_data)

        # r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # # ---------------------------------#
        # #   将图像转到HSV上
        # # ---------------------------------#
        # hue, sat, val = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        # dtype = image_data.dtype
        # # ---------------------------------#
        # #   应用变换
        # # ---------------------------------#
        # x = np.arange(0, 256, dtype=r.dtype)
        # lut_hue = ((x * r[0]) % 180).astype(dtype)
        # lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        # lut_val = np.clip(x * r[2], 0, 255).astype(dtype)
        #
        # image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        # image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        # ---------------------------------#
        #   对真实框进行调整
        # ---------------------------------#
        if len(box) > 0:
            np.random.shuffle(box)
            box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
            box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
            # if flip: box[:, [0, 2]] = w - box[:, [2, 0]]
            box[:, 0:2][box[:, 0:2] < 0] = 0
            box[:, 2][box[:, 2] > w] = w
            box[:, 3][box[:, 3] > h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w > 1, box_h > 1)]

        return image_data, box, pose

    def merge_bboxes(self, bboxes, cutx, cuty):
        merge_bbox = []
        for i in range(len(bboxes)):
            for box in bboxes[i]:
                tmp_box = []
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]

                if i == 0:
                    if y1 > cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 1:
                    if y2 < cuty or x1 > cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x2 = cutx

                if i == 2:
                    if y2 < cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y1 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx

                if i == 3:
                    if y1 > cuty or x2 < cutx:
                        continue
                    if y2 >= cuty and y1 <= cuty:
                        y2 = cuty
                    if x2 >= cutx and x1 <= cutx:
                        x1 = cutx
                tmp_box.append(x1)
                tmp_box.append(y1)
                tmp_box.append(x2)
                tmp_box.append(y2)
                tmp_box.append(box[-1])
                merge_bbox.append(tmp_box)
        return merge_bbox

    def get_random_data_with_Mosaic(self, annotation_line, input_shape, jitter=0.3, hue=.1, sat=0.7, val=0.4):
        h, w = input_shape
        min_offset_x = self.rand(0.3, 0.7)
        min_offset_y = self.rand(0.3, 0.7)

        image_datas = []
        box_datas = []
        index = 0
        for line in annotation_line:
            # ---------------------------------#
            #   每一行进行分割
            # ---------------------------------#
            line_content = line.split()
            # ---------------------------------#
            #   打开图片
            # ---------------------------------#
            image = Image.open(line_content[0])
            image = cvtColor(image)

            # ---------------------------------#
            #   图片的大小
            # ---------------------------------#
            iw, ih = image.size
            # ---------------------------------#
            #   保存框的位置
            # ---------------------------------#
            box = np.array([np.array(list(map(int, box.split(',')))) for box in line_content[1:]])

            # ---------------------------------#
            #   是否翻转图片
            # ---------------------------------#
            flip = self.rand() < .5
            if flip and len(box) > 0:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                box[:, [0, 2]] = iw - box[:, [2, 0]]

            # ------------------------------------------#
            #   对图像进行缩放并且进行长和宽的扭曲
            # ------------------------------------------#
            new_ar = iw / ih * self.rand(1 - jitter, 1 + jitter) / self.rand(1 - jitter, 1 + jitter)
            scale = self.rand(.4, 1)
            if new_ar < 1:
                nh = int(scale * h)
                nw = int(nh * new_ar)
            else:
                nw = int(scale * w)
                nh = int(nw / new_ar)
            image = image.resize((nw, nh), Image.BICUBIC)

            # -----------------------------------------------#
            #   将图片进行放置，分别对应四张分割图片的位置
            # -----------------------------------------------#
            if index == 0:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y) - nh
            elif index == 1:
                dx = int(w * min_offset_x) - nw
                dy = int(h * min_offset_y)
            elif index == 2:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y)
            elif index == 3:
                dx = int(w * min_offset_x)
                dy = int(h * min_offset_y) - nh

            new_image = Image.new('RGB', (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image)

            index = index + 1
            box_data = []
            # ---------------------------------#
            #   对box进行重新处理
            # ---------------------------------#
            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]
                box_data = np.zeros((len(box), 5))
                box_data[:len(box)] = box

            image_datas.append(image_data)
            box_datas.append(box_data)

        # ---------------------------------#
        #   将图片分割，放在一起
        # ---------------------------------#
        cutx = int(w * min_offset_x)
        cuty = int(h * min_offset_y)

        new_image = np.zeros([h, w, 3])
        new_image[:cuty, :cutx, :] = image_datas[0][:cuty, :cutx, :]
        new_image[cuty:, :cutx, :] = image_datas[1][cuty:, :cutx, :]
        new_image[cuty:, cutx:, :] = image_datas[2][cuty:, cutx:, :]
        new_image[:cuty, cutx:, :] = image_datas[3][:cuty, cutx:, :]

        new_image = np.array(new_image, np.uint8)
        # ---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        # ---------------------------------#
        r = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        # ---------------------------------#
        #   将图像转到HSV上
        # ---------------------------------#
        hue, sat, val = cv2.split(cv2.cvtColor(new_image, cv2.COLOR_RGB2HSV))
        dtype = new_image.dtype
        # ---------------------------------#
        #   应用变换
        # ---------------------------------#
        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        new_image = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        new_image = cv2.cvtColor(new_image, cv2.COLOR_HSV2RGB)

        # ---------------------------------#
        #   对框进行进一步的处理
        # ---------------------------------#
        new_boxes = self.merge_bboxes(box_datas, cutx, cuty)

        return new_image, new_boxes

    def get_random_data_with_MixUp(self, image_1, box_1, image_2, box_2):
        new_image = np.array(image_1, np.float32) * 0.5 + np.array(image_2, np.float32) * 0.5
        if len(box_1) == 0:
            new_boxes = box_2
        elif len(box_2) == 0:
            new_boxes = box_1
        else:
            new_boxes = np.concatenate([box_1, box_2], axis=0)
        return new_image, new_boxes

    def rotate_cam(self, image, t, q, bbox, R_change):
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
        h = self.K @ R_change.T @ np.linalg.inv(self.K)

        height, width = image.shape[:2]
        # Update pose
        t_new = R_change.T @ t
        puv = self.pinhole_model(t_new)
        q_change = Quaternion(matrix=R_change)
        q_new = (q_change.conjugate * Quaternion(q)).q

        vertices_b = np.array([[-0.79, -0.3, -0.38],
                               [-0.79, -0.3, 0.4],
                               [-0.3, 0.3, -0.38],
                               [-0.3, 0.3, 0.4],
                               [0.79, -0.3, -0.38],
                               [0.79, -0.3, 0.4],
                               [0.35, 0.3, -0.38],
                               [0.35, 0.3, 0.4]]) * 0.4

        vertices_c = Quaternion(q_new).rotation_matrix @ vertices_b.transpose() + t_new[np.newaxis].repeat(8,
                                                                                                           axis=0).transpose()
        vertices_uv = self.pinhole_model(vertices_c).transpose()

        x1, y1 = tuple(vertices_uv[:, :-1].min(axis=0))
        x2, y2 = tuple(vertices_uv[:, :-1].max(axis=0))

        bbox_new = np.array([[x1, y1, x2, y2, 0]])

        warp_flag = (0 < puv[0] < width) and (0 < puv[1] < height)
        # warp_flag = (0 < x1 < x2 < width) and (0 < y1 < y2 < height)


        if not warp_flag:
            return image, t, q, bbox
        else:
            image_warped = cv2.warpPerspective(image, h, (width, height))

            return image_warped, t_new, q_new, bbox_new

    def pinhole_model(self, t):
        p = self.K @ t
        return p / p[-1]


# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    poses = []
    for img, box, pose in batch:
        images.append(img)
        bboxes.append(box)
        poses.append(pose)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    poses = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in poses]
    return images, bboxes, poses
