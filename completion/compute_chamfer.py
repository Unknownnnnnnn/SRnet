import torch
import numpy as np
import torch.utils.data as data
import h5py
import os
import random
from vis_utils import plot_single_pcd, plot_single_pcd_red, plot_single_pcd_blue, plot_single_pcd_all, plot_single_pcd_res, plot_single_pcd2, plot_single_pcd3, plot_single_pcd_all2
from model_utils import *
sys.path.append("../utils")
import pygicp
from mm3d_pn2 import three_interpolate, furthest_point_sample, gather_points, grouping_operation, grouping_operation
def random_pose(max_angle, max_trans):
    R = random_rotation(max_angle)
    t = random_translation(max_trans)
    return np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)

import numpy as np
from sklearn.neighbors import NearestNeighbors
def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''


    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t


def nearest_neighbor(src, dst):
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''


    neigh = NearestNeighbors(n_neighbors=1)
    neigh.fit(dst)
    distances, indices = neigh.kneighbors(src, return_distance=True)
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.00001, gt = None, loop = True):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

#     assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]
    if (loop == False):
        gt = gt.reshape((4,4))
    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0]))
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T)
    dst[:m,:] = np.copy(B.T)
    
    # apply the initial pose estimation
    if init_pose is not None:
        src = np.dot(init_pose, src)
    
    prev_error = 0
    src_ = src
    for i in range(max_iterations):
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T)

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T)

        # update the current source
        if (loop == False):
            gt_src = np.dot(gt, src_)
        src = np.dot(T, src)
        
        # check error
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error
    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T)
    if (loop == True):
        return T, distances, i, src
    else:
        return T, distances, i

def random_rotation(max_angle):
    axis = np.random.randn(3)
    axis /= np.linalg.norm(axis)
    angle = np.random.rand() * max_angle
    A = np.array([[0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * A + (1 - np.cos(angle)) * np.dot(A, A)
    return R


def random_translation(max_dist):
    t = np.random.randn(3)
    t /= np.linalg.norm(t)
    t *= np.random.rand() * max_dist
    return np.expand_dims(t, 1)


 
 
 
def noise_Gaussian(points, std):
    noise = np.random.normal(0, std, points.shape)
    out = points + noise
    return out
 

class MVP_CP(data.Dataset):
    def __init__(self, prefix="train"):
        if prefix=="train":
            self.file_path = './data/MVP_Train_CP.h5'
        elif prefix=="val":
            self.file_path = './data/MVP_Test_CP.h5'
        elif prefix=="test":
            self.file_path = './data/MVP_Test_CP.h5'
        else:
            raise ValueError("ValueError prefix should be [train/val/test] ")

        self.prefix = prefix

        input_file = h5py.File(self.file_path, 'r')
        self.input_data = np.array(input_file['incomplete_pcds'][()])
        self.gt_data = np.array(input_file['complete_pcds'][()])
        self.labels = np.array(input_file['labels'][()])

        input_file.close()
        self.len = self.input_data.shape[0]
        self.max_angle = 1 / 180 * np.pi
        self.max_trans = 0.01
        
        print(self.labels)
        
        
        
        
        
        
#         input_file = h5py.File('./data/MVP_Test_CP.h5', 'r')
#         self.input_data = np.array(input_file['incomplete_pcds'][()])
#         self.gt_data = np.array(input_file['complete_pcds'][()])
#         self.labels = np.array(input_file['labels'][()])
        
#         partial = self.input_data[0]
#         sys_error = np.random.normal(0, 0.001, (self.input_data.shape[0], self.input_data.shape[2]))
#         target_np_noised = noise_Gaussian(self.input_data[0], 0.005)
#         partial = partial + sys_error[0][np.newaxis, :]
#         partial = partial[np.newaxis,:,:]
#         from tqdm import tqdm
        
#         for i in tqdm(range(self.input_data.shape[0]-1)):
#             now = self.input_data[i+1]
#             now = noise_Gaussian(now, 0.005)
#             now_points = (now + sys_error[i+1][np.newaxis, :])[np.newaxis, :]
#             partial = np.concatenate((partial, now_points), 0)
#         input_file_ = h5py.File('./data/MVP_Test3_CP.h5', 'w')
#         input_file_['incomplete_pcds'] = partial
#         input_file_['complete_pcds'] = self.gt_data
#         input_file_['labels'] = self.labels
        
#         print(dsadsaD)
        
        
        
        
        
        
        
        
#         mean1 = 0.
#         mean2 = 0.
#         mean3 = 0.
#         mean4 = 0.
#         for i in range(len(self.gt_data)):
#             a = torch.from_numpy(self.gt_data[i]).unsqueeze(0).transpose(2,1).cuda()
#             pred_center = a
# #             p_idx = furthest_point_sample(a.transpose(2,1).contiguous(), 2048)   #0
# #             pred_center = gather_points(a.contiguous(), p_idx.int())
#             b = torch.abs(knn_dist(pred_center, 5))
#             b = b.max(-1)[0].squeeze().mean()
#             mean1 += b
#             p_idx = furthest_point_sample(a.transpose(2,1).contiguous(), 512)   #0
#             pred_center = gather_points(a.contiguous(), p_idx.int())
#             b = torch.abs(knn_dist(pred_center, 32))
#             b = b.max(-1)[0].squeeze().mean()
#             mean2 += b
#             p_idx = furthest_point_sample(a.transpose(2,1).contiguous(), 256)   #0
#             pred_center = gather_points(a.contiguous(), p_idx.int())
#             b = torch.abs(knn_dist(pred_center, 64))
#             b = b.max(-1)[0].squeeze().mean()
#             mean3 += b
#             p_idx = furthest_point_sample(a.transpose(2,1).contiguous(), 128)   #0
#             pred_center = gather_points(a.contiguous(), p_idx.int())
#             b = torch.abs(knn_dist(pred_center, 64))
#             b = b.max(-1)[0].squeeze().mean()
#             mean4 += b
#         print(mean1 / (len(self.gt_data)))
#         print(mean2 / (len(self.gt_data)))
#         print(mean3 / (len(self.gt_data)))
#         print(mean4 / (len(self.gt_data)))
#         print(dsadsadsa)
        
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        import open3d as o3d
        if (self.prefix == 'train'):
            sequence_dir = os.path.join('/home/lq/素材/chamfer/GICP')
            filelist = os.listdir(sequence_dir)
            filelist = range(len(filelist))
            filelist = [str(i) for i in filelist]
            num_frames = len(filelist)
            scan_paths = {}
            for i in range(num_frames):
                now_path = sequence_dir + '/' + filelist[i] + '.pcd'
                scan_paths[i] = now_path
            
            sequence_dir = os.path.join('/home/lq/素材/chamfer/25 - Cloud.pcd')
#             filelist = os.listdir(sequence_dir)
#             filelist = range(95)
#             filelist = [str(i) for i in filelist]
#             num_frames = len(filelist)
#             scan_paths2 = {}
#             for i in range(num_frames):
#                 now_path = sequence_dir + '/' + filelist[i] + '.pcd'
#                 scan_paths2[i] = now_path
            import matplotlib.pyplot as plt
            from matplotlib.animation import FFMpegWriter
            import numpy as np
            import matplotlib
            from matplotlib.ticker import MaxNLocator
            font2 = {'family' : 'Times New Roman',

            'weight' : 'normal',

            'size' : 20,

            }
            matplotlib.rcParams.update({'text.color' : "white",
                     'axes.labelcolor' : "white", 'xtick.color':'white', 'ytick.color':'white','xtick.labelsize':'14',})
            
            fig_idx = 0
            fig = plt.figure(fig_idx, facecolor = 'black')
            writer = FFMpegWriter(fps=15)
            video_name = "./GICP.mp4"
            aaa = []
            num_frames_to_skip_to_show = 1
            num_frames_to_save = np.floor(95/1)
            with writer.saving(fig, video_name, num_frames_to_save):
                for i in range(len(scan_paths)):
                    src = o3d.io.read_point_cloud(scan_paths[i])
                    tgt = o3d.io.read_point_cloud(sequence_dir)
                    src = torch.from_numpy(np.asarray(src.points)).unsqueeze(0).cuda().float()
                    tgt = torch.from_numpy(np.asarray(tgt.points)).unsqueeze(0).cuda().float()


                    _, _, f1 = calc_cd(src, tgt)
                    print(aaa)
                    aaa.append(f1.detach().cpu().numpy()[0])
                    
                    plt.clf()
                    ax = fig.add_subplot(111)
                    ax.patch.set_facecolor('black')
                    ax.plot(np.array(aaa), color='red',linewidth=2) # kitti camera coord for clarity
                    xlim = plt.xlim()
                    max_xlim = max(map(abs, xlim))
                    plt.xlim((0, 20))
                    ylim = plt.ylim()
                    max_ylim = max(map(abs, ylim))
                    plt.ylim((0, 1))
                    plt.tick_params(labelsize=14)
                    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

#                     plt.gca().yaxis.set_major_locator(MaxNLocator(integer=True))
                    plt.xlabel('Frame',font2)
                    plt.ylabel('Modeling precision',font2)
                    plt.draw()
                    fig.savefig("./gif.png")
                    plt.pause(0.01) #is necessary for the plot to update for some reason
                    writer.grab_frame()
            
            
            print(dsads)
            
            
            
            
            
            
            
            
            print(sdadsad)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
        if (self.prefix == 'val'):
            ori_index = (index // 26) *26  #  0   , 26
            tar_index = (index // 26) *26 +26   # 26    , 52
            random_list = [index, ((index + 1) % 26) + ori_index, ((index + 2) % 26) + ori_index, ((index + 3) % 26) + ori_index]
            partial = np.concatenate((self.input_data[random_list[0]],self.input_data[random_list[1]],self.input_data[random_list[2]],self.input_data[random_list[3]]))
            partial = torch.from_numpy(partial)
            
        if self.prefix == "train":
            complete = torch.from_numpy((self.gt_data[index // 26]))
            label = (self.labels[index])
            return label, partial, complete
        elif self.prefix =='val':
            complete = torch.from_numpy((self.gt_data[index // 26]))
            label = (self.labels[index])
            return label, partial, complete
a = MVP_CP()
a.__getitem__(0)