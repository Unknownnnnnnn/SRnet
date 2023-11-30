import open3d as o3d
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import sys
sys.path.append("../utils")
import torch
from model_utils import *
from mm3d_pn2 import three_interpolate, furthest_point_sample, gather_points, grouping_operation, grouping_operation
def plot_single_pcds(points, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('auto')
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    rotation_matrix = np.asarray([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    pcd = pcd.transform(rotation_matrix)
    
    
    
    X, Y, Z = get_pts(pcd)
    
    points_array = np.concatenate([X[:,np.newaxis], Y[:,np.newaxis], Z[:,np.newaxis]], -1)
    
    print(points_array.shape)
    
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points_array)
    
    
    o3d.io.write_point_cloud('./fpss.pcd', pcd)
    
    t = Z
    ax.scatter(X, Y, Z, c=t, cmap='jet', marker='o', s=0.5, linewidths=0)
    ax.grid(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    set_axes_equal(ax)
    plt.axis('off')
    plt.savefig(save_path, format='png', dpi=1200)
    plt.close()  

def get_pts(pcd):
    points = np.asarray(pcd.points)
    X = []
    Y = []
    Z = []
    for pt in range(points.shape[0]):
        X.append(points[pt][0])
        Y.append(points[pt][1])
        Z.append(points[pt][2])
    return np.asarray(X), np.asarray(Y), np.asarray(Z)


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_single_pcd(points, save_path, res = False):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('auto')
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if (res == True):
        print(np.array(pcd.points).shape)
        pcd = pcd.remove_radius_outlier(nb_points=5, radius=0.1)
        pcd = pcd[0]
        print(np.array(pcd.points).shape)
    rotation_matrix = np.asarray([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    pcd = pcd.transform(rotation_matrix)
    
    X, Y, Z = get_pts(pcd)
    
    points_array = np.concatenate([X[:,np.newaxis], Y[:,np.newaxis], Z[:,np.newaxis]], -1)
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points_array)
    
    o3d.io.write_point_cloud(save_path.replace('png','pcd'), pcd)
    
    t = Z
    ax.scatter(X, Y, Z, cmap='jet', marker='o', s=1.5, linewidths=0)
    ax.grid(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    set_axes_equal(ax)
    plt.axis('off')
    plt.savefig(save_path, format='png', dpi=1200)
    plt.close()

def plot_single_pcd2(points, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('auto')
    a = torch.from_numpy(points).unsqueeze(0).contiguous().cuda().float()
    p_idx = furthest_point_sample(a, 2048).detach().cpu().numpy()
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    rotation_matrix = np.asarray([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    pcd = pcd.transform(rotation_matrix)
    
    X, Y, Z = get_pts(pcd)
    
    points_array = np.concatenate([X[:,np.newaxis], Y[:,np.newaxis], Z[:,np.newaxis]], -1)
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points_array)
    
    
    o3d.io.write_point_cloud(save_path.replace('png','pcd'), pcd)
    
    t = Z
    
    ax.scatter(X[p_idx[0]], Y[p_idx[0]], Z[p_idx[0]], marker='o', s=0.5, linewidths=0)
#     ax.scatter(X, Y, Z, c='gainsboro', marker='o', s=0.5, linewidths=0)
    ax.grid(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    set_axes_equal(ax)
    plt.axis('off')
    plt.savefig(save_path, format='png', dpi=1200)
    plt.close()
def plot_single_pcd3(points, save_path):
    c = torch.from_numpy(points).unsqueeze(0).cuda()
    
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('auto')
    
    
    a = torch.from_numpy(points).unsqueeze(0).contiguous().cuda()
    p_idx = furthest_point_sample(a, 32).detach().cpu().numpy()
    
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    rotation_matrix = np.asarray([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    pcd = pcd.transform(rotation_matrix)
    
    X, Y, Z = get_pts(pcd)
    
    points_array = np.concatenate([X[:,np.newaxis], Y[:,np.newaxis], Z[:,np.newaxis]], -1)
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points_array)
    
    
    o3d.io.write_point_cloud(save_path.replace('png','pcd'), pcd)
    
    t = Z
    
    a1 = torch.from_numpy(points[4096:6144,:]).unsqueeze(0)
    a2 = torch.from_numpy(points).unsqueeze(0)
    dist, idx = knn_point(8, a2, a1)
    a = dist.mean(-1)
    idx = np.argsort(a)[:,:224]
    idx = idx.detach().cpu().numpy()
    
    
    ax.scatter(X[p_idx[0]], Y[p_idx[0]], Z[p_idx[0]],c='r', marker='.', s=7.5, linewidths=0)
    ax.scatter(X[idx[0]], Y[idx[0]], Z[idx[0]],c='r', marker='.', s=7.5, linewidths=0)
    ax.scatter(X, Y, Z, c='gainsboro', marker='o', s=0.5, linewidths=0)

    ax.grid(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    set_axes_equal(ax)
    plt.axis('off')
    plt.savefig(save_path, format='png', dpi=1200)
    plt.close()

def plot_single_pcd_red(points, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('auto')
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    rotation_matrix = np.asarray([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    pcd = pcd.transform(rotation_matrix)
    
    X, Y, Z = get_pts(pcd)
    
    points_array = np.concatenate([X[:,np.newaxis], Y[:,np.newaxis], Z[:,np.newaxis]], -1)
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points_array)
    
    
#     o3d.io.write_point_cloud(save_path.replace('png','pcd'), pcd)
    
    t = Z
    ax.scatter(X, Y, Z, c='r', cmap='jet', marker='o', s=0.5, linewidths=0)
    ax.grid(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    set_axes_equal(ax)
    plt.axis('off')
    plt.savefig(save_path, format='png', dpi=1200)
    plt.close()
def plot_single_pcd_blue(points, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('auto')
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    rotation_matrix = np.asarray([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    pcd = pcd.transform(rotation_matrix)
    
    X, Y, Z = get_pts(pcd)
    
    points_array = np.concatenate([X[:,np.newaxis], Y[:,np.newaxis], Z[:,np.newaxis]], -1)
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points_array)
    
    
#     o3d.io.write_point_cloud(save_path.replace('png','pcd'), pcd)
    
    t = Z
    ax.scatter(X, Y, Z, c='b', cmap='jet', marker='o', s=0.5, linewidths=0)
    ax.grid(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    set_axes_equal(ax)
    plt.axis('off')
    plt.savefig(save_path, format='png', dpi=1200)
    plt.close()

def plot_single_pcd_all(points1, points2, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('auto')
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points1))
    rotation_matrix = np.asarray([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    pcd = pcd.transform(rotation_matrix)
    
    X, Y, Z = get_pts(pcd)
    
    points_array = np.concatenate([X[:,np.newaxis], Y[:,np.newaxis], Z[:,np.newaxis]], -1)
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points_array)
    
    t = Z
    ax.scatter(X, Y, Z, c='r', cmap='jet', marker='o', s=0.5, linewidths=0)
    
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points2))
    rotation_matrix = np.asarray([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    pcd = pcd.transform(rotation_matrix)
    
    X2, Y2, Z2 = get_pts(pcd)
    
    points_array = np.concatenate([X2[:,np.newaxis], Y2[:,np.newaxis], Z2[:,np.newaxis]], -1)
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points_array)
    
    t = Z2
    ax.scatter(X2, Y2, Z2, c='b', cmap='jet', marker='o', s=0.5, linewidths=0)
    
    
    ax.grid(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    set_axes_equal(ax)
    plt.axis('off')
    plt.savefig(save_path, format='png', dpi=1200)
    plt.close()
def plot_single_pcd_all2(points1, points2, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('auto')
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points1))
    rotation_matrix = np.asarray([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    pcd = pcd.transform(rotation_matrix)
    
    X, Y, Z = get_pts(pcd)
    
    points_array = np.concatenate([X[:,np.newaxis], Y[:,np.newaxis], Z[:,np.newaxis]], -1)
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points_array)
    
    t = Z
    ax.scatter(X, Y, Z, c='r', cmap='jet', marker='o', s=0.5, linewidths=0)
    
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points2))
    rotation_matrix = np.asarray([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    pcd = pcd.transform(rotation_matrix)
    
    X2, Y2, Z2 = get_pts(pcd)
    
    points_array = np.concatenate([X2[:,np.newaxis], Y2[:,np.newaxis], Z2[:,np.newaxis]], -1)
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points_array)
    
    t = Z2
    ax.scatter(X2, Y2, Z2, c='b', cmap='jet', marker='o', s=0.5, linewidths=0)
    
    a = np.array([X[0],X2[1]])
    b = np.array([Y[0],Y2[1]])
    c = np.array([Z[0],Z2[1]])
    ax.plot(a,b,c, c='green', linestyle='--', marker='o',markersize=1.2, linewidth=0.9)
    
    
    ax.grid(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    set_axes_equal(ax)
    plt.axis('off')
    plt.savefig(save_path, format='png', dpi=1200)
    plt.close()

def plot_single_pcd_res(points1, points2, save_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('auto')
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points1))
    rotation_matrix = np.asarray([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    pcd = pcd.transform(rotation_matrix)
    
    X, Y, Z = get_pts(pcd)
    
    points_array = np.concatenate([X[:,np.newaxis], Y[:,np.newaxis], Z[:,np.newaxis]], -1)
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points_array)
    
    t = Z
    ax.scatter(X, Y, Z, c='r', cmap='jet', marker='o', s=0.5, linewidths=0)
    
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points2))
    rotation_matrix = np.asarray([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
    pcd = pcd.transform(rotation_matrix)
    
    X2, Y2, Z2 = get_pts(pcd)
    
    points_array = np.concatenate([X2[:,np.newaxis], Y2[:,np.newaxis], Z2[:,np.newaxis]], -1)
    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points_array)
    
    t = Z2
    ax.scatter(X2, Y2, Z2, c='b', cmap='jet', marker='o', s=0.5, linewidths=0)
    
    ax.grid(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    set_axes_equal(ax)
    plt.axis('off')
    plt.savefig(save_path, format='png', dpi=1200)
    plt.close()


