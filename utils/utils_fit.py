import os

import torch
from tqdm import tqdm
import numpy as np

from utils.utils import get_lr
import utils.se3lib as se3lib

# def recoverXYZ(self, bbox, offset, depth):
#     K = torch.tensor(
#         [[1744.92206139719, 0, 737.272795902663], [0, 1746.58640701753, 528.471960188736], [0, 0, 1]]).cuda()
#     b, num_anchors, _ = bbox.shape
#     p = torch.concat(((bbox[..., 0:2] + offset), torch.ones_like(depth)), dim=-1)
#     P = torch.matmul(torch.linalg.inv(K).repeat(b * num_anchors, 1, 1), (depth.repeat(1, 1, 3) * p).view(-1, 3, 1))
#     return P.view(b, num_anchors, 3)


# def decode_outputs(outputs, input_shape):
#     grids = []
#     strides = []
#     hw = [x.shape[-2:] for x in outputs]
#     # ---------------------------------------------------#
#     #   outputs输入前代表每个特征层的预测结果
#     #   batch_size, 4 + 1 + num_classes, 80, 80 => batch_size, 4 + 1 + num_classes, 6400
#     #   batch_size, 5 + num_classes, 40, 40
#     #   batch_size, 5 + num_classes, 20, 20
#     #   batch_size, 4 + 1 + num_classes, 6400 + 1600 + 400 -> batch_size, 4 + 1 + num_classes, 8400
#     #   堆叠后为batch_size, 8400, 5 + num_classes
#     # ---------------------------------------------------#
#     outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)
#     # ---------------------------------------------------#
#     #   获得每一个特征点属于每一个种类的概率
#     # ---------------------------------------------------#
#     outputs[:, :, 4:6] = torch.sigmoid(outputs[:, :, 4:6])
#     for h, w in hw:
#         # ---------------------------#
#         #   根据特征层的高宽生成网格点
#         # ---------------------------#
#         grid_y, grid_x = torch.meshgrid([torch.arange(h), torch.arange(w)])
#         # ---------------------------#
#         #   1, 6400, 2
#         #   1, 1600, 2
#         #   1, 400, 2
#         # ---------------------------#
#         grid = torch.stack((grid_x, grid_y), 2).view(1, -1, 2)
#         shape = grid.shape[:2]
#
#         grids.append(grid)
#         strides.append(torch.full((shape[0], shape[1], 1), input_shape[0] / h))
#     # ---------------------------#
#     #   将网格点堆叠到一起
#     #   1, 6400, 2
#     #   1, 1600, 2
#     #   1, 400, 2
#     #
#     #   1, 8400, 2
#     # ---------------------------#
#     grids = torch.cat(grids, dim=1).type(outputs.type())
#     strides = torch.cat(strides, dim=1).type(outputs.type())
#     # ------------------------#
#     #   根据网格点进行解码
#     # ------------------------#
#     outputs[..., :2] = (outputs[..., :2] + grids) * strides
#     outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
#
#
#     outputs[:, :, 6:9] = recoverXYZ(outputs[..., :2], outputs[:, :, 6:8], outputs[:, :, 8:9])
#     gt_r = se3lib.compute_rotation_matrix_from_quaternion(target)
#     est_r = se3lib.symmetric_orthogonalization(pred)
#
#     # -----------------#
#     #   归一化
#     # -----------------#
#     # outputs[..., [0, 2]] = outputs[..., [0, 2]] / input_shape[1]
#     # outputs[..., [1, 3]] = outputs[..., [1, 3]] / input_shape[0]
#     return outputs

def pose_err(est_pose, gt_pose):
    pass


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

    # stats = np.zeros((len(gen_val), 3))

    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        images, targets, poses = batch[0], batch[1], batch[2]
        with torch.no_grad():
            if cuda:
                images = images.cuda(local_rank)
                targets = [ann.cuda(local_rank) for ann in targets]
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

            # outputs = decode_outputs(outputs, list(images.shape[-2:]))
            #
            # # Evaluate error
            # posit_err, rel_posit_err, orient_err = pose_err(est_pose, gt_pose)
            #
            # # Collect statistics
            # stats[val_samples - batch_size:val_samples, 0] = posit_err.cpu().numpy()
            # stats[val_samples - batch_size:val_samples, 1] = orient_err.cpu().numpy()
            # stats[val_samples - batch_size:val_samples, 2] = rel_posit_err.cpu().numpy()
            #
            # pbar.desc = "Pose error: {:.3f}[m], {:.3f}[deg]".format(
            #     posit_err.mean().item(), orient_err.mean().item())

        val_loss += loss_value.item()
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1)})
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
