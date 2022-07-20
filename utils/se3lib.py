import math
from os import error
import numpy as np
import torch
import tqdm
import time

import matplotlib.pyplot as plt

from scipy.spatial.transform import Rotation as R
from pyquaternion import Quaternion


rotation_dimension_map = {'svd': 9, 'ortho6d': 6, 'ortho5d': 5, 'quaternion': 4, 'axisAngle': 3, 'euler': 3}

"""
Numpy functions
"""

def basic_rotation(ang, unit='deg', axis='x'):
    "https://en.wikipedia.org/wiki/Rotation_matrix object rotation, not the frame"
    if unit == 'deg':
        ang = np.deg2rad(ang)
    elif unit == 'rad':
        pass
    else:
        raise error("angle unit error")

    if axis == 'x':
        R = np.array([[1,           0,            0],
                      [0, np.cos(ang), -np.sin(ang)],
                      [0, np.sin(ang),  np.cos(ang)]])
    elif axis == 'y':
        R = np.array([[ np.cos(ang), 0, np.sin(ang)],
                      [           0, 1,           0],
                      [-np.sin(ang), 0, np.cos(ang)]])
    elif axis == 'z':
        R = np.array([[np.cos(ang), -np.sin(ang), 0],
                      [np.sin(ang),  np.cos(ang), 0],
                      [          0,            0, 1]])
    else:
        raise error("rotation axis error")

    return R


def euler2SO3(roll, pitch, yaw, unit="deg"):
    """ Convert euler angles in degrees to a rotation matrix using XYZ order (valid)"""

    R = basic_rotation(yaw, unit, axis="z") @ basic_rotation(pitch, unit, axis="y") @ basic_rotation(roll, unit, axis="x")

    return R

def euler2SO3_left(pitch, yaw, roll):
    """ Convert euler angles in degrees to a rotation matrix using XYZ order (valid)"""
    cos_pitch = np.cos(pitch*np.pi/180)
    sin_pitch = np.sin(pitch*np.pi/180)
    cos_yaw = np.cos(yaw*np.pi/180)
    sin_yaw = np.sin(yaw*np.pi/180)
    cos_roll = np.cos(roll*np.pi/180)
    sin_roll = np.sin(roll*np.pi/180)

    R = np.matrix([[cos_yaw*cos_roll, sin_pitch*sin_yaw*cos_roll - cos_pitch*sin_roll, cos_pitch*sin_yaw*cos_roll + sin_pitch*sin_roll],
        [cos_yaw*sin_roll, sin_pitch*sin_yaw*sin_roll + cos_pitch*cos_roll, cos_pitch*sin_yaw*sin_roll - sin_pitch*cos_roll],
        [-sin_yaw, sin_pitch*cos_yaw, cos_pitch*cos_yaw]])

    return R

def quat2SO3(q):
    q = q /np.linalg.norm(q)
    w, x, y, z = tuple(q[i] for i in range(4))
    row1 = np.array([[1-2*y**2-2*z**2,     2*x*y-2*z*w,     2*x*z+2*y*w]])
    row2 = np.array([[    2*x*y+2*z*w, 1-2*x**2-2*z**2,     2*y*z-2*x*w]])
    row3 = np.array([[    2*x*z-2*y*w,     2*y*z+2*x*w, 1-2*x**2-2*y**2]])
    R = np.concatenate((row1, row2, row3), axis=0)

    return R


def SO32quat(R):
    """ 
    https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    Shuster, 1997 - A survey of attitude representations
    """

    # R trace
    # trR = R[0,0] + R[1,1] + R[2,2]

    qw_abs = np.sqrt( R[0,0] + R[1,1] + R[2,2] + 1) / 2.
    qx_abs = np.sqrt( R[0,0] - R[1,1] - R[2,2] + 1) / 2.
    qy_abs = np.sqrt(-R[0,0] + R[1,1] - R[2,2] + 1) / 2.
    qz_abs = np.sqrt(-R[0,0] - R[1,1] + R[2,2] + 1) / 2.

    tmp = max(qw_abs, qx_abs, qy_abs, qz_abs)

    if tmp == qw_abs:
        qw = tmp
        qx = (R[2,1] - R[1,2])/(4*tmp)
        qy = (R[0,2] - R[2,0])/(4*tmp)
        qz = (R[1,0] - R[0,1])/(4*tmp)
    elif tmp == qx_abs:
        qx = tmp
        qy = (R[0,1] + R[1,0])/(4*tmp)
        qz = (R[0,2] + R[2,0])/(4*tmp)
        qw = (R[2,1] - R[1,2])/(4*tmp)
    elif tmp == qy_abs:
        qy = tmp
        qx = (R[0,1] + R[1,0])/(4*tmp)
        qz = (R[1,2] + R[2,1])/(4*tmp)
        qw = (R[0,2] - R[2,0])/(4*tmp)
    else:
        qz = tmp
        qx = (R[0,2] + R[2,0])/(4*tmp)
        qy = (R[1,2] + R[2,1])/(4*tmp)
        qw = (R[1,0] - R[0,1])/(4*tmp)
    
    sgn = np.sign(qw)
    q = sgn * np.array([qw, qx, qy, qz])
    return q


def SO32quat_batch(matrices):
    """
    https://en.wikipedia.org/wiki/Rotation_matrix#Quaternion
    Shuster, 1997 - A survey of attitude representations
    """

    batch = matrices.shape[0]
    qw_abs = np.sqrt(1.0 + matrices[:, 0, 0] + matrices[:, 1, 1] + matrices[:, 2, 2]) / 2.0
    qx_abs = np.sqrt(1.0 + matrices[:, 0, 0] - matrices[:, 1, 1] - matrices[:, 2, 2]) / 2.0
    qy_abs = np.sqrt(1.0 - matrices[:, 0, 0] + matrices[:, 1, 1] - matrices[:, 2, 2]) / 2.0
    qz_abs = np.sqrt(1.0 - matrices[:, 0, 0] - matrices[:, 1, 1] + matrices[:, 2, 2]) / 2.0

    quats = np.zeros((batch, 4))
    for idx in range(batch):
        tmp = max(qw_abs[idx], qx_abs[idx], qy_abs[idx], qz_abs[idx])

        if tmp == qw_abs[idx]:
            qw = tmp
            qx = (matrices[idx, 2, 1] - matrices[idx, 1, 2]) / (4 * tmp)
            qy = (matrices[idx, 0, 2] - matrices[idx, 2, 0]) / (4 * tmp)
            qz = (matrices[idx, 1, 0] - matrices[idx, 0, 1]) / (4 * tmp)
        elif tmp == qx_abs[idx]:
            qx = tmp
            qy = (matrices[idx, 0, 1] + matrices[idx, 1, 0]) / (4 * tmp)
            qz = (matrices[idx, 0, 2] + matrices[idx, 2, 0]) / (4 * tmp)
            qw = (matrices[idx, 2, 1] - matrices[idx, 1, 2]) / (4 * tmp)
        elif tmp == qy_abs[idx]:
            qy = tmp
            qx = (matrices[idx, 0, 1] + matrices[idx, 1, 0]) / (4 * tmp)
            qz = (matrices[idx, 1, 2] + matrices[idx, 2, 1]) / (4 * tmp)
            qw = (matrices[idx, 0, 2] - matrices[idx, 2, 0]) / (4 * tmp)
        else:
            qz = tmp
            qx = (matrices[idx, 0, 2] + matrices[idx, 2, 0]) / (4 * tmp)
            qy = (matrices[idx, 1, 2] + matrices[idx, 2, 1]) / (4 * tmp)
            qw = (matrices[idx, 1, 0] - matrices[idx, 0, 1]) / (4 * tmp)

        sgn = np.sign(qw)
        q = sgn * np.array([qw, qx, qy, qz])
        quats[idx, :] = q / np.linalg.norm(q)
    return quats

def quat_mult_2(a, b):
    """ Multiply 2 quaternions """

    # c = np.matrix([[a[0], -a[1], -a[2], -a[3]],
    #                [a[1],  a[0],  a[3], -a[2]],
    #                [a[2], -a[3],  a[0],  a[1]],
    #                [a[3],  a[2], -a[1],  a[0]]])

    c = np.array([[a[0], -a[1], -a[2], -a[3]],
                  [a[1],  a[0], -a[3],  a[2]],
                  [a[2],  a[3],  a[0], -a[1]],
                  [a[3], -a[2],  a[1],  a[0]]])

    result = np.dot(c,b)

    # Enforcing quaternion unit for sanity
    result = result / np.linalg.norm(result)

    return result




def quat2axisAngle(y):
    theta = np.linalg.norm(y)

    if theta < eps:
        quat = np.array([1, 0, 0, 0])
    else:
        v = y / theta
        q0 = np.array([np.cos(theta/2.)])
        qv = v*np.sin(theta/2.)
        quat = np.concatenate((q0, qv))
    return quat




# axis angle y = \theta v where theta is angle of rotation about axis v
# y = logm(R) where R is a rotation matrix. I use Rodriguez' rotation formula
# Shuster (102-103)
def rotationMatrix2axisAngle(R):
    tR = 0.5 * (np.trace(R) - 1)
    theta = np.arccos(np.clip(tR, -1., 1.))
    tmp = 0.5 * (R - R.T)
    n = np.array([tmp[2, 1], tmp[0, 2], tmp[1, 0]])
    if np.linalg.norm(n) > eps:
        n = n / np.linalg.norm(n)
    else:
        n = np.zeros(3)
    v = theta * n
    return v


# function to get rotation matrix given axis-angle representation
def axisAngle2rotationMatrix(v):
    theta = np.linalg.norm(v)
    if theta < eps:
        R = np.eye(3)
    else:
        n = v / theta
        n_skew = np.array(
            [[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])
        R = np.eye(3) + np.sin(theta) * n_skew + \
            (1 - np.cos(theta)) * np.dot(n_skew, n_skew)
    return R


# get rotation matrix R(az, el, ct) given the three euler angles :
# azimuth az, elevation el, camera-tilt ct
def euler2rotationMatrix(az, el, ct):
    ca = np.cos(np.radians(az))
    sa = np.sin(np.radians(az))
    cb = np.cos(np.radians(el))
    sb = np.sin(np.radians(el))
    cc = np.cos(np.radians(ct))
    sc = np.sin(np.radians(ct))
    Ra = np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    Rb = np.array([[1, 0, 0], [0, cb, -sb], [0, sb, cb]])
    Rc = np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    R = np.dot(np.dot(Rc, Rb), Ra)
    return R





"""
Pytorch functions
"""

############################################## utils ##############################################
# batch*n (valid)
def normalize_vector(v):
    # batch = v.shape[0]
    # v_mag = torch.sqrt(v.pow(2).sum(1))  # batch
    # v_mag = torch.max(v_mag, torch.autograd.Variable(
    #     torch.FloatTensor([1e-8]).cuda()))
    # v_mag = v_mag.view(batch, 1).expand(batch, v.shape[1])
    # v_norm = v / v_mag
    v_norm = torch.nn.functional.normalize(v, p=2, dim=1)
    return v_norm

# u, v batch*3 (valid, equiv to torch.cross(u,v))
def cross_product(u, v):
    if u.shape == v.shape:
        batch = u.shape[0]

        i = u[:, 1] * v[:, 2] - u[:, 2] * v[:, 1]
        j = u[:, 2] * v[:, 0] - u[:, 0] * v[:, 2]
        k = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]

        out = torch.cat((i.view(batch, 1), j.view(batch, 1),
                        k.view(batch, 1)), 1)  # batch*3
        return out
    else:
        raise error("inputs sizes dont match")

# u,a batch*3 (valid)
def proj_u_a(u, a):
    batch = u.shape[0]
    top = u[:, 0] * a[:, 0] + u[:, 1] * a[:, 1] + u[:, 2] * a[:, 2]
    bottom = u[:, 0] * u[:, 0] + u[:, 1] * u[:, 1] + u[:, 2] * u[:, 2]
    bottom = torch.max(torch.autograd.Variable(
        torch.zeros(batch).cuda()) + 1e-8, bottom)
    factor = (top / bottom).view(batch, 1).expand(batch, 3)
    out = factor * u
    return out



##################################################################################################



################################ compute rotation matrix from ... ################################

def compute_rotation_matrix_from_euler(euler):
    
    batch = euler.shape[0]

    c1 = torch.cos(euler[:, 0]).view(batch, 1)  # batch*1
    s1 = torch.sin(euler[:, 0]).view(batch, 1)  # batch*1
    c2 = torch.cos(euler[:, 1]).view(batch, 1)  # batch*1
    s2 = torch.sin(euler[:, 1]).view(batch, 1)  # batch*1
    c3 = torch.cos(euler[:, 2]).view(batch, 1)  # batch*1
    s3 = torch.sin(euler[:, 2]).view(batch, 1)  # batch*1


    # in the rotation order of ZXZ
    # https://en.wikipedia.org/wiki/Euler_angles
    row1 = torch.cat((c1*c3-s1*c2*s3, s1 * c3 + c1 *
                     c2 * s3, s2 * s3), 1).view(-1, 1, 3)  # batch*1*3
    row2 = torch.cat((- c1 * s3 - s1 * c2 * c3, - s1 * s3 +
                     c1 * c2 * c3, s2 * c3), 1).view(-1, 1, 3)  # batch*1*3
    row3 = torch.cat((s1 * s2, - c1 * s2, c2), 1).view(-1, 1, 3)  # batch*1*3

    matrix = torch.cat((row1, row2, row3), 1)  # batch*3*3

    return matrix


def compute_rotation_matrix_from_quaternion(quaternion):

    batch = quaternion.shape[0]

    quat = normalize_vector(quaternion)

    qw = quat[..., 0].view(batch, 1)
    qx = quat[..., 1].view(batch, 1)
    qy = quat[..., 2].view(batch, 1)
    qz = quat[..., 3].view(batch, 1)

    # Unit quaternion rotation matrices computatation
    xx = qx * qx
    yy = qy * qy
    zz = qz * qz
    xy = qx * qy
    xz = qx * qz
    yz = qy * qz
    xw = qx * qw
    yw = qy * qw
    zw = qz * qw

    row0 = torch.cat((1 - 2 * yy - 2 * zz, 2 * xy - 2 *
                     zw, 2 * xz + 2 * yw), 1)  # batch*3
    row1 = torch.cat((2 * xy + 2 * zw, 1 - 2 * xx - 2 *
                     zz, 2 * yz - 2 * xw), 1)  # batch*3
    row2 = torch.cat((2 * xz - 2 * yw, 2 * yz + 2 * xw,
                     1 - 2 * xx - 2 * yy), 1)  # batch*3

    matrix = torch.cat((row0.view(batch, 1, 3), row1.view(
        batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3

    return matrix



def stereographic_project(u):
    """
        stereographic projection: decreases dimension by one.
        """
    batch_size = u.shape[0]
    v = normalize_vector(u)
    Pu = v[:, 1:]/(1-v[:, 0]).view(batch_size,1).repeat(1, u.shape[1]-1)
    return Pu

def stereographic_unproject(u):
    """
        Inverse of stereographic projection: increases dimension by one.
        """
    batch_size = u.shape[0]
    u_norm = torch.pow(u, 2).sum(1).sqrt() # a = tensor([1.0042])
    Qu = torch.autograd.Variable(torch.zeros(
        batch_size, u.shape[1] + 1))
    Qu[:, 0] = (u_norm.pow(2)-1) / (2*u_norm) # Qu1 = tensor([0.0042])
    Qu[:, 1:] = u / u_norm.view(-1,1).repeat(1,3) # tensor([[0.6436, 0.6280, 0.4375]])
    return Qu

def compute_rotation_matrix_from_ortho5d(u):
    r6d = compute_r6d_from_r5d(u)
    r = compute_rotation_matrix_from_ortho6d(r6d)
    ensure_SO3(r.cpu())
    return r

def compute_r6d_from_r5d(u):
    batch_size = u.shape[0]
    Qu = stereographic_unproject(u[:, 2:5])
    r6d = torch.cat((u[:, 0:2], Qu.cuda()), 1)
    return r6d

def compute_r5d_from_r6d(gamma):
    batch_size = gamma.shape[0]
    Pu = stereographic_project(gamma[:, 2:6])
    r5d = torch.cat((gamma[:, 0:2], Pu), 1)
    return r5d


# get rotation matrix R given the r5d vector
# def r5d2rotationMatrix(r5d):
#     sin_cos = r5d[0:2]
#     sin_cos_mag = torch.max(torch.sqrt(sin_cos.pow(2).sum(1)),
#                             torch.autograd.Variable(torch.DoubleTensor([1e-8]).cuda()))  # batch
#     sin_cos_mag = sin_cos_mag.view(batch, 1).expand(batch, 2)  # batch*2
#     sin_cos = sin_cos / sin_cos_mag  # batch*2
#
#     axis = r5d[:, 2:5]  # batch*3
#     axis_mag = torch.max(torch.sqrt(axis.pow(2).sum(1)),
#                          torch.autograd.Variable(torch.DoubleTensor([1e-8]).cuda()))  # batch
#
#     axis_mag = axis_mag.view(batch, 1).expand(batch, 3)  # batch*3
#     axis = axis / axis_mag  # batch*3
#     out_rotation = torch.cat((sin_cos, axis), 1)  # batch*5
#
#     return out_rotation


def symmetric_orthogonalization(x):
    """Maps 9D input vectors onto SO(3) via symmetric orthogonalization.

    x: shoud have size [batch_size, 9]

    Output hase size [batch_size, 3, 3], where each inner 3x3 matrix is in SO(3).
    """
    m = x.view(-1, 3, 3)
    u, s, v = torch.svd(m)
    vt = torch.transpose(v, 1, 2)
    det = torch.det(torch.matmul(u, vt))
    det = det.view(-1, 1, 1)
    vt = torch.cat((vt[:, :2, :], vt[:, -1:, :] * det), 1)
    r = torch.matmul(u, vt)
    return r





# poses batch*6
# poses
def compute_rotation_matrix_from_ortho6d(poses):
    x_raw = poses[:, 0:3]  # batch*3
    y_raw = poses[:, 3:6]  # batch*3

    x = normalize_vector(x_raw)  # batch*3
    z = cross_product(x, y_raw)  # batch*3
    z = normalize_vector(z)  # batch*3
    y = cross_product(z, x)  # batch*3

    x = x.view(-1, 3, 1)
    y = y.view(-1, 3, 1)
    z = z.view(-1, 3, 1)
    matrix = torch.cat((x, y, z), 2)  # batch*3*3
    return matrix






# function to get rotation matrix given axis-angle representation
def compute_quaternions_from_axisAngle(axisAngle):
    batch = axisAngle.shape[0]
    theta = torch.sqrt(axisAngle.pow(2).sum(1))  # batch
    theta = torch.max(theta, torch.autograd.Variable(
        torch.FloatTensor([1e-8]).cuda()))
    v = axisAngle / theta.view(batch, 1).expand(batch, axisAngle.shape[1])
    cos = torch.cos(theta/2)
    sin = torch.sin(theta/2)

    qs = cos.view(batch, 1)
    qv = v * sin.view(batch, 1).expand(batch, axisAngle.shape[1])
    quat = torch.cat((qs, qv), 1)
    return quat


def compute_rotation_matrix_from_axisAngle(axisAngle):
    quat = compute_quaternions_from_axisAngle(axisAngle)
    matrix = compute_rotation_matrix_from_quaternion(quat)
    return matrix


def ensure_SO3(r):
    eps = 1e-6*r.shape[0]
    rt = r.transpose(1,2)
    rrt = torch.bmm(r, rt)
    i = torch.eye(3,3).unsqueeze(dim=0).repeat(r.shape[0], 1, 1)
    err1 = torch.dist(rrt, i, 2).item()
    det = torch.det(r)
    err2 = torch.dist(det, torch.ones(r.shape[0]), 2).item()
    assert err1 < eps and err2 < eps



# matrices batch*3*3
# both matrix are orthogonal rotation matrices
# out theta between 0 to 180 degree batch
def compute_geodesic_distance_from_two_matrices(m1, m2):
    assert m1.shape[1:] == torch.Size([3, 3]) and m2.shape[1:] == torch.Size([3, 3])
    # ensure_SO3(m1.cpu())
    batch = m1.shape[0]

    m = torch.bmm(m1, m2.transpose(1, 2))  # batch*3*3

    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = torch.min(cos, torch.autograd.Variable(torch.ones(batch).cuda()))
    cos = torch.max(cos, torch.autograd.Variable(torch.ones(batch).cuda()) * -1)

    theta = torch.acos(cos)

    return theta


def get_sampled_rotation_matrices_by_quat(batch):
    # quat = torch.autograd.Variable(torch.rand(batch,4).cuda())
    quat = torch.autograd.Variable(torch.randn(batch, 4).cuda())
    matrix = compute_rotation_matrix_from_quaternion(quat)
    return matrix


# axisAngle batch*3*3s angle, x,y,z
def get_sampled_rotation_matrices_by_axisAngle(batch):
    theta = torch.autograd.Variable(
        torch.FloatTensor(np.random.uniform(-1, 1, batch) * np.pi).cuda())  # [0, pi] #[-180, 180]
    sin = torch.sin(theta)
    axis = torch.autograd.Variable(torch.randn(batch, 3).cuda())
    axis = normalize_vector(axis)  # batch*3
    qw = torch.cos(theta)
    qx = axis[:, 0] * sin
    qy = axis[:, 1] * sin
    qz = axis[:, 2] * sin

    # Unit quaternion rotation matrices computatation
    xx = (qx * qx).view(batch, 1)
    yy = (qy * qy).view(batch, 1)
    zz = (qz * qz).view(batch, 1)
    xy = (qx * qy).view(batch, 1)
    xz = (qx * qz).view(batch, 1)
    yz = (qy * qz).view(batch, 1)
    xw = (qx * qw).view(batch, 1)
    yw = (qy * qw).view(batch, 1)
    zw = (qz * qw).view(batch, 1)

    row0 = torch.cat((1 - 2*yy - 2*zz,     2*xy - 2*zw,     2*xz + 2*yw), 1)  # batch*3
    row1 = torch.cat((    2*xy + 2*zw, 1 - 2*xx - 2*zz,     2*yz - 2*xw), 1)  # batch*3
    row2 = torch.cat((    2*xz - 2*yw,     2*yz + 2*xw, 1 - 2*xx - 2*yy), 1)  # batch*3

    matrices = torch.cat((row0.view(batch, 1, 3), row1.view(
        batch, 1, 3), row2.view(batch, 1, 3)), 1)  # batch*3*3

    return matrices


# input batch*3*3
# output batch*3 (x, y, z in radiant)
# the rotation is in the sequence of x,y,z
# def compute_euler_angles_from_rotation_matrices(matrices):
#     batch = matrices.shape[0]
    
#     sy = torch.sqrt(matrices[:, 0, 0] * matrices[:, 0, 0] + matrices[:, 1, 0] * matrices[:, 1, 0])
#     singular = sy < 1e-6
#     singular = singular.float()

#     x = torch.atan2(matrices[:, 1, 0], matrices[:, 0, 0])  # alpha Z yaw
#     y = torch.atan2(-matrices[:, 2, 0], sy)                # beta  Y pitch
#     z = torch.atan2(matrices[:, 2, 1], matrices[:, 2, 2])  # gamma X roll

#     xs = torch.atan2(-matrices[:, 1, 2], matrices[:, 1, 1])
#     ys = torch.atan2(-matrices[:, 2, 0], sy)
#     zs = matrices[:, 1, 0] * 0

#     euler = torch.autograd.Variable(torch.zeros(batch, 3).cuda())
#     euler[:, 0] = x * (1 - singular) + xs * singular
#     euler[:, 1] = y * (1 - singular) + ys * singular
#     euler[:, 2] = z * (1 - singular) + zs * singular
#     euler[:, 2] = x * (1 - singular) + xs * singular
#     euler[:, 1] = y * (1 - singular) + ys * singular
#     euler[:, 0] = z * (1 - singular) + zs * singular

#     return euler


# input batch*4
# output batch*4 (valid)
# def compute_quaternions_from_axisAngles(axisAngles):
#     w = torch.cos(axisAngles[:, 0] / 2)
#     sin = torch.sin(axisAngles[:, 0] / 2)
#     x = sin * axisAngles[:, 1]
#     y = sin * axisAngles[:, 2]
#     z = sin * axisAngles[:, 3]

#     quat = torch.cat((w.view(-1, 1), x.view(-1, 1),
#                      y.view(-1, 1), z.view(-1, 1)), 1)

#     return quat


# quaternions batch*4,
# matrices batch*3*3
def compute_quaternions_from_rotation_matrices(matrices):
    batch = matrices.shape[0]

    qw_abs = torch.nan_to_num(torch.sqrt(1.0 + matrices[:, 0, 0] + matrices[:, 1, 1] + matrices[:, 2, 2]) / 2.0)
    qx_abs = torch.nan_to_num(torch.sqrt(1.0 + matrices[:, 0, 0] - matrices[:, 1, 1] - matrices[:, 2, 2]) / 2.0)
    qy_abs = torch.nan_to_num(torch.sqrt(1.0 - matrices[:, 0, 0] + matrices[:, 1, 1] - matrices[:, 2, 2]) / 2.0)
    qz_abs = torch.nan_to_num(torch.sqrt(1.0 - matrices[:, 0, 0] - matrices[:, 1, 1] + matrices[:, 2, 2]) / 2.0)

    quats = torch.zeros((batch, 4)).cuda()
    for idx in range(batch):
        tmp = max(qw_abs[idx], qx_abs[idx], qy_abs[idx], qz_abs[idx])

        if tmp == qw_abs[idx]:
            qw = tmp
            qx = (matrices[idx, 2, 1] - matrices[idx, 1, 2]) / (4 * tmp)
            qy = (matrices[idx, 0, 2] - matrices[idx, 2, 0]) / (4 * tmp)
            qz = (matrices[idx, 1, 0] - matrices[idx, 0, 1]) / (4 * tmp)
        elif tmp == qx_abs[idx]:
            qx = tmp
            qy = (matrices[idx, 0, 1] + matrices[idx, 1, 0]) / (4 * tmp)
            qz = (matrices[idx, 0, 2] + matrices[idx, 2, 0]) / (4 * tmp)
            qw = (matrices[idx, 2, 1] - matrices[idx, 1, 2]) / (4 * tmp)
        elif tmp == qy_abs[idx]:
            qy = tmp
            qx = (matrices[idx, 0, 1] + matrices[idx, 1, 0]) / (4 * tmp)
            qz = (matrices[idx, 1, 2] + matrices[idx, 2, 1]) / (4 * tmp)
            qw = (matrices[idx, 0, 2] - matrices[idx, 2, 0]) / (4 * tmp)
        else:
            qz = tmp
            qx = (matrices[idx, 0, 2] + matrices[idx, 2, 0]) / (4 * tmp)
            qy = (matrices[idx, 1, 2] + matrices[idx, 2, 1]) / (4 * tmp)
            qw = (matrices[idx, 1, 0] - matrices[idx, 0, 1]) / (4 * tmp)

        sgn = torch.sign(qw)
        q = sgn * torch.tensor([qw, qx, qy, qz]).cuda()
        quats[idx, :] = q / q.norm()

    return quats

def quat_weighted_avg(Q, W):
    """Compute the average quaternion q of a set of quaternions Q,
    based on a Linear Least Squares Solution of the form: Ax = 0

    The sum of squared dot products between quaternions:
        L(q) = sum_i w_i(Q_i^T*q)^T(Q_i^T*q)^T

    achieves its maximum when its derivative wrt q is zero, i.e.:

        Aq = 0 where A = sum_i (Q_i*Q_i^T)

    Therefore, the optimal q is simply the right null space of A.

    For more solutions check:
    Markley, F. Landis, et al. "Averaging quaternions." Journal of Guidance, Control, and Dynamics (2007)

    Arguments:
        Q: The set of quaternions
        W: The respective weights
    Returns:
        q_avg: The solution
        H_inv: The uncertainty in the maximum likelihood sense

    """

    N = np.size(Q,0)

    # Compute A
    A = np.zeros(shape=(4, 4), dtype=np.float32)
    for i in range(N):
        a = np.matrix([Q[i, 0], Q[i, 1], Q[i, 2], Q[i, 3]])
        A += a.transpose() * a * W[i]

    s, v = np.linalg.eig(A)
    idx = np.argsort(s)

    q_avg = v[:, idx[-1]] # 0

    # Due to numerical errors, we need to enforce normalization
    q_avg = q_avg / np.linalg.norm(q_avg)

    H_inv = np.linalg.inv(A)

    return q_avg, H_inv


if __name__ == '__main__':
    r = R.from_euler('z', 0, degrees=True)
    print(r.as_matrix())

    print(r.as_quat())

    # R_1 = np.array([[0.9734767, 0.06202098, 0.22021925],
    #        [0.06198093, -0.9980521, 0.00709831],
    #        [0.22023053, 0.00673936, -0.9754246]], dtype=np.float32)

    # R_1_tensor = torch.from_numpy(R_1).unsqueeze(0)

    # q = compute_quaternions_from_rotation_matrices(R_1_tensor)

    # R = compute_rotation_matrix_from_quaternion(q)
    # print(q)

    # print(a_new)

    # count = 0
    # for _ in tqdm.tqdm(range(int(1e4))):

    #     pitch = np.random.random(2)
    #     yaw = np.random.random(2)
    #     roll = np.random.random(2)
    #     R0 = euler2SO3(roll[0], pitch[0], yaw[0], unit="rad")
    #     R1 = euler2SO3(roll[1], pitch[1], yaw[1], unit="rad")

    #     q0 = SO32quat(R0)
    #     q1 = SO32quat(R1)

    #     q2 = quat_mult_2(q0,q1)
    #     # print(R)

    #     R_0 = SO3.RPY((roll[0], pitch[0], yaw[0]), unit='rad', order='zyx')
    #     R_1 = SO3.RPY((roll[1], pitch[1], yaw[1]), unit='rad', order='zyx')
    #     q_0 = UQ(R_0)
    #     q_1 = UQ(R_1)

    #     q_2 = q_0 * q_1

    #     delta = np.linalg.norm(q2-q_2.A)

    #     if delta < eps:
    #         count += 1
    #     else:
    #         pass

    # print(count)

    # tmp = rotationMatrix2axisAngle(R_1.R)
    # myq_1 = quat2axisAngle(tmp)
    # print('myq1:', myq_1)

    # quat2SO3(myq_1)

    # R_2 = SO3.RPY((roll[1], pitch[1], yaw[1]), unit='deg', order='zyx')
    # q_2 = UQ(R_2)

    # print('q2:', q_2.A)
    # tmp = rotationMatrix2axisAngle(R_2.R)
    # myq_2 = quat2axisAngle(tmp)
    # print('myq2:', myq_2)

    # q = q_1 * q_2

    # myq = quat_mult(myq_1, myq_2)

    # print(q)
    # # euler = torch.tensor([pitch, yaw, roll]).view(1,3)

    # # R_1 = compute_rotation_matrix_from_euler(euler)

    # # R_2 = SO3.Rx(pitch, unit='deg') * SO3.Ry(yaw, unit='deg') * SO3.Rz(roll, unit='deg')

    # # R_3 = SO3.Rz(roll, unit='deg') * SO3.Ry(yaw, unit='deg') * SO3.Rx(pitch, unit='deg')

    # # quat = SO32quat(R)
    # # print(R_1)
    # array([[0.9734767, 0.06202098, 0.22021925],
    #        [0.06198093, -0.9980521, 0.00709831],
    #        [0.22023053, 0.00673936, -0.9754246]], dtype=float32)

    # count = 0
    # for _ in tqdm.tqdm(range(int(1e4))):

    #     pitch = np.random.random(2)
    #     yaw = np.random.random(2)
    #     roll = np.random.random(2)

    #     R_0 = SO3.RPY((roll[0], pitch[0], yaw[0]), unit='rad', order='zyx')
    #     R_1 = SO3.RPY((roll[1], pitch[1], yaw[1]), unit='rad', order='zyx')
    #     q_0 = UQ(R_0)
    #     q_1 = UQ(R_1)

    #     theta = compute_geodesic_distance_from_two_matrices(torch.from_numpy(R_0.R).unsqueeze(0), torch.from_numpy(R_1.R).unsqueeze(0))

    #     # R = symmetric_orthogonalization(torch.from_numpy(R_0.R).unsqueeze(0))

    #     # theta_q = 2*np.arccos(np.dot(q_0.A, q_1.A))

    #     # delta = np.linalg.norm(theta-theta_q)

    #     if theta > 2*np.pi - theta:
    #         count += 1
