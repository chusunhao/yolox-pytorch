import os

import torch
from tqdm import tqdm
import numpy as np
from utils.utils_bbox import decode_outputs, non_max_suppression

from utils.utils import get_lr
import utils.se3lib as se3lib


# Evaluation utils
##########################
def pose_err(est_pose, gt_pose):
    """
    Calculate the position and orientation error given the estimated and ground truth pose(s
    :param est_pose: (torch.Tensor) a batch of estimated poses (Nx12, N is the batch size)
    :param gt_pose: (torch.Tensor) a batch of ground-truth poses (Nx7, N is the batch size)
    :return: position error(s) and orientation errors(s)
    """
    gt_p = gt_pose[:, 0:3]
    est_p = est_pose[:, 0:3]
    posit_err = torch.norm(est_p - gt_p, dim=1)
    rel_posit_err = posit_err / torch.norm(gt_p, dim=1)

    gt_r = se3lib.compute_rotation_matrix_from_quaternion(gt_pose[:, 3:])
    est_r = est_pose[:, 3:].view(-1,3,3)
    theta = se3lib.compute_geodesic_distance_from_two_matrices(gt_r, est_r)
    orient_err = theta * 180 / np.pi

    return posit_err, rel_posit_err, orient_err

def calculate_iou(pred, target):
    assert pred.shape[0] == target.shape[0]

    tl = torch.max(
        (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
    )
    br = torch.min(
        (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
    )

    area_p = torch.prod(pred[:, 2:], 1)
    area_g = torch.prod(target[:, 2:], 1)

    en = (tl < br).type(tl.type()).prod(dim=1)
    area_i = torch.prod(br - tl, 1) * en
    area_u = area_p + area_g - area_i
    iou = (area_i) / (area_u + 1e-16)
    return iou


def fit_one_epoch(model_train, model, ema, yolo_loss, loss_history, eval_callback, optimizer, epoch, epoch_step,
                  epoch_step_val, gen, gen_val, Epoch, cuda, fp16, scaler, save_period, save_dir, local_rank=0):
    loss = 0
    val_loss = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break

        images, targets, poses = batch[0], batch[1], batch[2]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
                poses = [ann.cuda(local_rank) for ann in poses]
        # ----------------------#
        #   清零梯度
        # ----------------------#
        optimizer.zero_grad()
        if not fp16:
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = model_train(images)

            # ----------------------#
            #   计算损失
            # ----------------------#
            loss_value = yolo_loss(outputs, targets, poses)

            # ----------------------#
            #   反向传播
            # ----------------------#
            loss_value.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                outputs = model_train(images)
                # ----------------------#
                #   计算损失
                # ----------------------#
                loss_value = yolo_loss(outputs, targets, poses)

            # ----------------------#
            #   反向传播
            # ----------------------#
            scaler.scale(loss_value).backward()
            scaler.step(optimizer)
            scaler.update()
        if ema:
            ema.update(model_train)

        loss += loss_value.item()

        if local_rank == 0:
            pbar.set_postfix(**{'loss': loss / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    if ema:
        model_train_eval = ema.ema
    else:
        model_train_eval = model_train.eval()

    stats = {"iou": [],
             "posit_err": [],
             "rel_posit_err": [],
             "orient_err": []}

    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets, poses = batch[0], batch[1], batch[2]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
                poses = [ann.cuda(local_rank) for ann in poses]
            # ----------------------#
            #   清零梯度
            # ----------------------#
            optimizer.zero_grad()
            # ----------------------#
            #   前向传播
            # ----------------------#
            outputs = model_train_eval(images)

            # ----------------------#
            #   计算损失
            # ----------------------#
            loss_value = yolo_loss(outputs, targets, poses)

            outputs = decode_outputs(outputs, list(images.shape[-2:]))

            results = non_max_suppression(outputs, num_classes=1, input_shape=[640, 640],
                                          image_shape=[1080, 1440], letterbox_image=True, maxdet=1)

            # Evaluate error
            gt_bbox = torch.cat(targets)[:, :4]
            est_bbox = torch.tensor(np.array(results)).squeeze()[:, :4].cuda()
            iou = calculate_iou(est_bbox, gt_bbox)
            est_pose = torch.tensor(np.array(results)).squeeze()[:, -12:].cuda()
            gt_pose = torch.cat(poses)
            posit_err, rel_posit_err, orient_err = pose_err(est_pose, gt_pose)

            # Collect statistics
            stats["posit_err"].append(posit_err.cpu().numpy())
            stats["rel_posit_err"].append(rel_posit_err.cpu().numpy())
            stats["orient_err"].append(orient_err.cpu().numpy())


        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                "iou": np.mean(stats["iou"]),
                                "posit_err": np.mean(stats["posit_err"]),
                                "rel_posit_err": np.mean(stats["rel_posit_err"]),
                                "orient_err": np.mean(stats["orient_err"])})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train_eval)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (loss / epoch_step, val_loss / epoch_step_val))

        # -----------------------------------------------#
        #   保存权值
        # -----------------------------------------------#
        if ema:
            save_state_dict = ema.ema.state_dict()
        else:
            save_state_dict = model.state_dict()

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(save_state_dict, os.path.join(save_dir, "ep%03d-loss%.3f-val_loss%.3f.pth" % (
            epoch + 1, loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(save_state_dict, os.path.join(save_dir, "best_epoch_weights.pth"))

        torch.save(save_state_dict, os.path.join(save_dir, "last_epoch_weights.pth"))
