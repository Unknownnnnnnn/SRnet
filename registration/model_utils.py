import torch
import numpy as np
import torch.nn as nn

def torch_qmul(q1, q2):
    """
    Multiply quaternion(s) q2q1, rotate q1 first, rotate q2 second.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q1.shape[-1] == 4
    assert q2.shape[-1] == 4

    original_shape = q1.shape

    # Compute outer product
    terms = torch.bmm(q1.view(-1, 4, 1), q2.view(-1, 1, 4))

    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    return torch.stack((w, x, y, z), dim=1).view(original_shape)


def torch_qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def torch_quat2euler(q, order, epsilon=0):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == "xyz":
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == "yzx":
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == "zxy":
        x = torch.asin(torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == "xzy":
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == "yxz":
        x = torch.asin(torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == "zyx":
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    else:
        raise

    return torch.stack((x, y, z), dim=1).view(original_shape)


def torch_euler2quat(e, order):
    """
    Convert Euler angles to quaternions.
    """
    assert e.size()[-1] == 3

    original_shape = [e.size()[0], 4]

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    rx = torch.stack((torch.cos(x / 2), torch.sin(x / 2), torch.zeros_like(x).cuda(), torch.zeros_like(x).cuda()), dim=1)
    ry = torch.stack((torch.cos(y / 2), torch.zeros_like(y).cuda(), torch.sin(y / 2), torch.zeros_like(y).cuda()), dim=1)
    rz = torch.stack((torch.cos(z / 2), torch.zeros_like(z).cuda(), torch.zeros_like(z).cuda(), torch.sin(z / 2)), dim=1)

    result = None
    for coord in order:
        if coord == "x":
            r = rx
        elif coord == "y":
            r = ry
        elif coord == "z":
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = torch_qmul(result, r)

    # Reverse antipodal representation to have a non-negative "w"
    if order in ["xyz", "yzx", "zxy"]:
        result *= -1

    return result.reshape(original_shape)


def torch_quat2mat(pose):
    # Separate each quaternion value.
    q0, q1, q2, q3 = pose[:, 0], pose[:, 1], pose[:, 2], pose[:, 3]
    # Convert quaternion to rotation matrix.
    # Ref: 	http://www-evasion.inrialpes.fr/people/Franck.Hetroy/Teaching/ProjetsImage/2007/Bib/besl_mckay-pami1992.pdf
    # A method for Registration of 3D shapes paper by Paul J. Besl and Neil D McKay.
    R11 = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
    R12 = 2 * (q1 * q2 - q0 * q3)
    R13 = 2 * (q1 * q3 + q0 * q2)
    R21 = 2 * (q1 * q2 + q0 * q3)
    R22 = q0 * q0 + q2 * q2 - q1 * q1 - q3 * q3
    R23 = 2 * (q2 * q3 - q0 * q1)
    R31 = 2 * (q1 * q3 - q0 * q2)
    R32 = 2 * (q2 * q3 + q0 * q1)
    R33 = q0 * q0 + q3 * q3 - q1 * q1 - q2 * q2
    R = torch.stack((torch.stack((R11, R12, R13), dim=0), torch.stack((R21, R22, R23), dim=0), torch.stack((R31, R32, R33), dim=0)), dim=0)

    rot_mat = R.permute((2, 0, 1))  # (B, 3, 3)
    translation = pose[:, 4:].unsqueeze(2)  # (B, 3, 1)
    transform = torch.cat((rot_mat, translation), dim=2)
    return transform  # (B, 3, 4)


def torch_transform_pose(pose_old, pose_new):
    quat_old, translate_old = pose_old[:, :4], pose_old[:, 4:]
    quat_new, translate_new = pose_new[:, :4], pose_new[:, 4:]

    quat = torch_qmul(quat_old, quat_new)
    translate = torch_qrot(quat_new, translate_old) + translate_new
    pose = torch.cat((quat, translate), dim=1)

    return pose


def torch_qinv(q):
    # expectes q in (w,x,y,z) format
    w = q[:, 0:1]
    v = q[:, 1:]
    inv = torch.cat([w, -v], dim=1)
    return inv


def torch_quat_rotate(point_cloud: torch.Tensor, pose_7d: torch.Tensor):
    ndim = point_cloud.dim()
    if ndim == 2:
        N, _ = point_cloud.shape
        assert pose_7d.shape[0] == 1
        # repeat transformation vector for each point in shape
        quat = pose_7d[:, 0:4].expand([N, -1])
        rotated_point_cloud = torch_qrot(quat, point_cloud)

    elif ndim == 3:
        B, N, _ = point_cloud.shape
        quat = pose_7d[:, 0:4].unsqueeze(1).expand([-1, N, -1]).contiguous()
        rotated_point_cloud = torch_qrot(quat, point_cloud)

    else:
        raise RuntimeError("point cloud dim must be 2 or 3 !")

    return rotated_point_cloud


def torch_quat_transform(pose_7d: torch.Tensor, pc: torch.Tensor, normal: torch.Tensor = None):
    pc_t = torch_quat_rotate(pc, pose_7d) + pose_7d[:, 4:].view(-1, 1, 3).repeat(1, pc.shape[1], 1)  # Ps" = R*Ps + t
    if normal is not None:
        normal_t = torch_quat_rotate(normal, pose_7d)
        return pc_t, normal_t
    else:
        return pc_t


def np_qmul(q, r):
    q = torch.from_numpy(q).contiguous()
    r = torch.from_numpy(r).contiguous()
    return torch_qmul(q, r).numpy()


def np_qrot(q, v):
    q = torch.from_numpy(q).contiguous()
    v = torch.from_numpy(v).contiguous()
    return torch_qrot(q, v).numpy()


def np_quat2euler(q, order, epsilon=0, use_gpu=False):
    if use_gpu:
        q = torch.from_numpy(q).cuda()
        return torch_quat2euler(q, order, epsilon).cpu().numpy()
    else:
        q = torch.from_numpy(q).contiguous()
        return torch_quat2euler(q, order, epsilon).numpy()


def np_qfix(q):
    """
    Enforce quaternion continuity across the time dimension by selecting
    the representation (q or -q) with minimal euclidean_distance (or, equivalently, maximal dot product)
    between two consecutive frames.
    Expects a tensor of shape (L, J, 4), where L is the sequence length and J is the number of joints.
    Returns a tensor of the same shape.
    """
    assert len(q.shape) == 3
    assert q.shape[-1] == 4

    result = q.copy()
    dot_products = np.sum(q[1:] * q[:-1], axis=2)
    mask = dot_products < 0
    mask = (np.cumsum(mask, axis=0) % 2).astype(bool)
    result[1:][mask] *= -1
    return result


def np_expmap2quat(e):
    """
    Convert axis-angle rotations (aka exponential maps) to quaternions.
    Stable formula from "Practical Parameterization of Rotations Using the Exponential Map".
    Expects a tensor of shape (*, 3), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 4).
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4
    e = e.reshape(-1, 3)

    theta = np.linalg.norm(e, axis=1).reshape(-1, 1)
    w = np.cos(0.5 * theta).reshape(-1, 1)
    xyz = 0.5 * np.sinc(0.5 * theta / np.pi) * e
    return np.concatenate((w, xyz), axis=1).reshape(original_shape)


def np_euler2quat(e, order):
    """
    Convert Euler angles to quaternions.
    """
    assert e.shape[-1] == 3

    original_shape = list(e.shape)
    original_shape[-1] = 4

    e = e.reshape(-1, 3)

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    rx = np.stack((np.cos(x / 2), np.sin(x / 2), np.zeros_like(x), np.zeros_like(x)), axis=1)
    ry = np.stack((np.cos(y / 2), np.zeros_like(y), np.sin(y / 2), np.zeros_like(y)), axis=1)
    rz = np.stack((np.cos(z / 2), np.zeros_like(z), np.zeros_like(z), np.sin(z / 2)), axis=1)

    result = None
    for coord in order:
        if coord == "x":
            r = rx
        elif coord == "y":
            r = ry
        elif coord == "z":
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = np_qmul(result, r)

    # Reverse antipodal representation to have a non-negative "w"
    if order in ["xyz", "yzx", "zxy"]:
        result *= -1

    return result.reshape(original_shape)
import numpy as np
import torch
# import transforms3d.euler as t3de
import transforms3d.quaternions as t3d
from scipy.spatial.transform import Rotation


def torch_identity(batch_size):
    return torch.eye(3, 4)[None, ...].repeat(batch_size, 1, 1)


def torch_inverse(g):
    """ Returns the inverse of the SE3 transform
    Args:
        g: (B, 3/4, 4) transform
    Returns:
        (B, 3, 4) matrix containing the inverse
    """
    # Compute inverse
    rot = g[..., 0:3, 0:3]
    trans = g[..., 0:3, 3]
    inverse_transform = torch.cat([rot.transpose(-1, -2), rot.transpose(-1, -2) @ -trans[..., None]], dim=-1)

    return inverse_transform


def torch_concatenate(a, b):
    """Concatenate two SE3 transforms,
    i.e. return a@b (but note that our SE3 is represented as a 3x4 matrix)
    Args:
        a: (B, 3/4, 4)
        b: (B, 3/4, 4)
    Returns:
        (B, 3/4, 4)
    """

    rot1 = a[..., :3, :3]
    trans1 = a[..., :3, 3]
    rot2 = b[..., :3, :3]
    trans2 = b[..., :3, 3]

    rot_cat = rot1 @ rot2
    trans_cat = rot1 @ trans2[..., None] + trans1[..., None]
    concatenated = torch.cat([rot_cat, trans_cat], dim=-1)

    return concatenated


def torch_transform(g, a, normals=None):
    """ Applies the SE3 transform
    Args:
        g: SE3 transformation matrix of size ([1,] 3/4, 4) or (B, 3/4, 4)
        a: Points to be transformed (N, 3) or (B, N, 3)
        normals: (Optional). If provided, normals will be transformed
    Returns:
        transformed points of size (N, 3) or (B, N, 3)
    """
    R = g[..., :3, :3]  # (B, 3, 3)
    p = g[..., :3, 3]  # (B, 3)

    if len(g.size()) == len(a.size()):
        b = torch.matmul(a, R.transpose(-1, -2)) + p[..., None, :]
    else:
        raise NotImplementedError
        b = R.matmul(a.unsqueeze(-1)).squeeze(-1) + p  # No batch. Not checked

    if normals is not None:
        rotated_normals = normals @ R.transpose(-1, -2)
        return b, rotated_normals

    else:
        return b

class LossL1(nn.Module):
    def __init__(self):
        super(LossL1, self).__init__()
        self.loss = nn.L1Loss()

    def __call__(self, input, target):
        return self.loss(input, target)


class LossL2(nn.Module):
    def __init__(self):
        super(LossL2, self).__init__()
        self.loss = nn.MSELoss()

    def __call__(self, input, target):
        return self.loss(input, target)


class LossCrossEntropy(nn.Module):
    def __init__(self, weight=None):
        super(LossCrossEntropy, self).__init__()
        self.loss = torch.nn.CrossEntropyLoss(weight=weight)

    def __call__(self, input, target, weight=None):
        return self.loss(input, target)
    
def torch_mat2quat(M):
    all_pose = []
    for i in range(M.size()[0]):
        rotate = M[i, :3, :3]
        translate = M[i, :3, 3]

        # Qyx refers to the contribution of the y input vector component to
        # the x output vector component.  Qyx is therefore the same as
        # M[0,1].  The notation is from the Wikipedia article.
        Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz = rotate.flatten()
        #     print(Qxx, Qyx, Qzx, Qxy, Qyy, Qzy, Qxz, Qyz, Qzz)
        # Fill only lower half of symmetric matrix
        K = torch.tensor([[Qxx - Qyy - Qzz, 0, 0, 0], [Qyx + Qxy, Qyy - Qxx - Qzz, 0, 0], [Qzx + Qxz, Qzy + Qyz, Qzz - Qxx - Qyy, 0],
                          [Qyz - Qzy, Qzx - Qxz, Qxy - Qyx, Qxx + Qyy + Qzz]]) / 3.0
        # Use Hermitian eigenvectors, values for speed
        vals, vecs = torch.symeig(K, True, False)
        # Select largest eigenvector, reorder to w,x,y,z quaternion

        q = vecs[[3, 0, 1, 2], torch.argmax(vals)].cuda()
        # Prefer quaternion with positive w
        # (q * -1 corresponds to same rotation as q)
        if q[0] < 0:
            q *= -1

        pose = torch.cat((q, translate), dim=0)
        all_pose.append(pose)
    all_pose = torch.stack(all_pose, dim=0)
    return all_pose  # (B, 7)


def np_identity():
    return np.eye(3, 4)


def np_transform(g: np.ndarray, pts: np.ndarray):
    """ Applies the SE3 transform
    Args:
        g: SE3 transformation matrix of size ([B,] 3/4, 4)
        pts: Points to be transformed ([B,] N, 3)
    Returns:
        transformed points of size (N, 3)
    """
    rot = g[..., :3, :3]  # (3, 3)
    trans = g[..., :3, 3]  # (3)

    transformed = pts[..., :3] @ np.swapaxes(rot, -1, -2) + trans[..., None, :]
    return transformed


def np_inverse(g: np.ndarray):
    """Returns the inverse of the SE3 transform
    Args:
        g: ([B,] 3/4, 4) transform
    Returns:
        ([B,] 3/4, 4) matrix containing the inverse
    """
    rot = g[..., :3, :3]  # (3, 3)
    trans = g[..., :3, 3]  # (3)

    inv_rot = np.swapaxes(rot, -1, -2)
    inverse_transform = np.concatenate([inv_rot, inv_rot @ -trans[..., None]], axis=-1)
    if g.shape[-2] == 4:
        inverse_transform = np.concatenate([inverse_transform, [[0.0, 0.0, 0.0, 1.0]]], axis=-2)

    return inverse_transform


def np_concatenate(a: np.ndarray, b: np.ndarray):
    """ Concatenate two SE3 transforms
    Args:
        a: First transform ([B,] 3/4, 4)
        b: Second transform ([B,] 3/4, 4)
    Returns:
        a*b ([B, ] 3/4, 4)
    """

    r_a, t_a = a[..., :3, :3], a[..., :3, 3]
    r_b, t_b = b[..., :3, :3], b[..., :3, 3]

    r_ab = r_a @ r_b
    t_ab = r_a @ t_b[..., None] + t_a[..., None]

    concatenated = np.concatenate([r_ab, t_ab], axis=-1)

    if a.shape[-2] == 4:
        concatenated = np.concatenate([concatenated, [[0.0, 0.0, 0.0, 1.0]]], axis=-2)

    return concatenated


def np_from_xyzquat(xyzquat):
    """Constructs SE3 matrix from x, y, z, qx, qy, qz, qw
    Args:
        xyzquat: np.array (7,) containing translation and quaterion
    Returns:
        SE3 matrix (4, 4)
    """
    rot = Rotation.from_quat(xyzquat[3:])
    trans = rot.apply(-xyzquat[:3])
    transform = np.concatenate([rot.as_dcm(), trans[:, None]], axis=1)
    transform = np.concatenate([transform, [[0.0, 0.0, 0.0, 1.0]]], axis=0)

    return transform


def np_mat2quat(transform):
    rotate = transform[:3, :3]
    translate = transform[:3, 3]
    quat = t3d.mat2quat(rotate)
    pose = np.concatenate([quat, translate], axis=0)
    return pose  # (7, )


def np_quat2mat(pose):
    # Separate each quaternion value.
    q0, q1, q2, q3 = pose[:, 0], pose[:, 1], pose[:, 2], pose[:, 3]
    # Convert quaternion to rotation matrix.
    # Ref: 	http://www-evasion.inrialpes.fr/people/Franck.Hetroy/Teaching/ProjetsImage/2007/Bib/besl_mckay-pami1992.pdf
    # A method for Registration of 3D shapes paper by Paul J. Besl and Neil D McKay.
    R11 = q0 * q0 + q1 * q1 - q2 * q2 - q3 * q3
    R12 = 2 * (q1 * q2 - q0 * q3)
    R13 = 2 * (q1 * q3 + q0 * q2)
    R21 = 2 * (q1 * q2 + q0 * q3)
    R22 = q0 * q0 + q2 * q2 - q1 * q1 - q3 * q3
    R23 = 2 * (q2 * q3 - q0 * q1)
    R31 = 2 * (q1 * q3 - q0 * q2)
    R32 = 2 * (q2 * q3 + q0 * q1)
    R33 = q0 * q0 + q3 * q3 - q1 * q1 - q2 * q2
    R = np.stack((np.stack((R11, R12, R13), axis=0), np.stack((R21, R22, R23), axis=0), np.stack((R31, R32, R33), axis=0)), axis=0)

    rot_mat = R.transpose((2, 0, 1))  # (B, 3, 3)
    translation = pose[:, 4:][:, :, None]  # (B, 3, 1)
    transform = np.concatenate((rot_mat, translation), axis=2)
    return transform  # (B, 3, 4)