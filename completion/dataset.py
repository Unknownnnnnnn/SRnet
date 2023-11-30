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
    axis = np.ones(3)#random.randn(3)
    axis /= np.linalg.norm(axis)
    angle =  max_angle
    A = np.array([[0, -axis[2], axis[1]],
                    [axis[2], 0, -axis[0]],
                    [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * A + (1 - np.cos(angle)) * np.dot(A, A)
    return R


def random_translation(max_dist):
    t = np.ones(3)#random.randn(3)
    t /= np.linalg.norm(t)
    t *=  max_dist
    return np.expand_dims(t, 1)

# def random_rotation(max_angle):
#     axis = np.random.randn(3)
#     axis /= np.linalg.norm(axis)
#     angle = np.random.rand() * max_angle
#     A = np.array([[0, -axis[2], axis[1]],
#                     [axis[2], 0, -axis[0]],
#                     [-axis[1], axis[0], 0]])
#     R = np.eye(3) + np.sin(angle) * A + (1 - np.cos(angle)) * np.dot(A, A)
#     return R


# def random_translation(max_dist):
#     t = np.random.randn(3)
#     t /= np.linalg.norm(t)
#     t *= np.random.rand() * max_dist
#     return np.expand_dims(t, 1)


 
 
 
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
        index = 0
        while (self.labels[index] != 6):
            index+=1
        index += 100
        while (self.labels[index] != 6):
            index+=1
#         index += 100
#         while (self.labels[index] != 1):
#             index+=1
#         index += 100
#         while (self.labels[index] != 1):
        list_ = range((index // 26) *26, (index // 26) *26 +26)
        if (self.prefix == 'train'):
            complete = torch.from_numpy((self.gt_data[index // 26]))
            random_list = np.array(list_)#random.sample(list_, 26)
#             for i in range(26):
#                 plot_single_pcd_red(self.input_data[random_list[i]].squeeze(),'./res/'+str(i)+'.png') 
#             print(dsads)
#             plot_single_pcd(complete,'./res/compelte.png')
            
#             a = noise_Gaussian(self.input_data[random_list[0]], 0.005)
#             transform = random_pose(1 / 180 * np.pi, 0.05)
#             a = a @ transform[:3, :3].T + transform[:3, 3]
#             plot_single_pcd(a,'./res/0.png')
#             pa = a
#             transform_save = transform
#             transform = transform[np.newaxis,:].reshape((-1,16))
#             for i in range(1,25):
#                 b = noise_Gaussian(self.input_data[random_list[i]], 0.005)
#                 transform2 = random_pose(1 / 180 * np.pi, 0.05)
#                 transform_save = np.matmul(transform_save, transform2)
#                 b = b @ transform_save[:3, :3].T + transform_save[:3, 3]
#                 plot_single_pcd(b,'./res/'+str(i)+'.png')
#                 transform = np.concatenate([transform, transform_save[np.newaxis,:].reshape((-1,16))])
#                 pa = np.concatenate([pa, b])
# #             pose_list = np.concatenate([transform[np.newaxis,:].reshape((-1,16)), transform2[np.newaxis,:].reshape((-1,16)), transform3[np.newaxis,:].reshape((-1,16)), transform4[np.newaxis,:].reshape((-1,16))])
# #             print(pose_list.shape)
#             filename = "./pose.csv"
#             np.savetxt(filename, transform, delimiter=",")
            
#             plot_single_pcd(pa,'./res/all.png')
            
#             print(dsada)
#             self.input_data[random_list[1]] = noise_Gaussian(self.input_data[random_list[1]], 0.01)
#             self.input_data[random_list[3]] = noise_Gaussian(self.input_data[random_list[3]], 0.005)
#             self.input_data[random_list[4]] = noise_Gaussian(self.input_data[random_list[4]], 0.005)
            
#             complete = torch.from_numpy((self.gt_data[index // 26]))
#             random_list = random.sample(list_, 4)
#             self.max_angle = 5 / 180 * np.pi
#             self.max_trans = 0.1
#             partial = np.concatenate((self.input_data[random_list[0]],self.input_data[random_list[1]],self.input_data[random_list[2]],self.input_data[random_list[3]]))
#             plot_single_pcd(partial,'./res/ori6.png')
#             for i in range(4):
#                 plot_single_pcd(self.input_data[random_list[i]],'./res/ori'+str(i)+'.png')
#             for i in range(4):
#                 transform = random_pose(self.max_angle, self.max_trans / 2)
#                 self.input_data[random_list[i]] = self.input_data[random_list[i]] @ transform[:3, :3].T + transform[:3, 3]
#                 plot_single_pcd(self.input_data[random_list[i]],'./res/ori_'+str(i)+'.png')
#             plot_single_pcd(complete,'./res/ori5.png')
#             print(dsadsad)
            
            
#             plot_single_pcd(self.input_data[random_list[0]],'./res/ori1.png')
#             plot_single_pcd(self.input_data[random_list[1]],'./res/ori2.png')
#             plot_single_pcd(self.input_data[random_list[2]],'./res/ori3.png')
#             plot_single_pcd(self.input_data[random_list[3]],'./res/ori4.png')
#             partial = np.concatenate((self.input_data[random_list[0]],self.input_data[random_list[1]],self.input_data[random_list[2]],self.input_data[random_list[3]]))
#             plot_single_pcd(partial,'./res/ori6.png')
#             plot_single_pcd(complete,'./res/ori5.png')
            
            
#             print(dsadsad)


#             transform = random_pose(self.max_angle, self.max_trans)
#             target_np_noised = noise_Gaussian(self.input_data[random_list[0]], 0.005)
#             a = (target_np_noised @ transform[:3, :3].T)+ transform[:3, 3]
#             plot_single_pcd(a.squeeze(),'./res/partial0.png')
#             for i in range(4):
#                 transform = random_pose(self.max_angle, self.max_trans)
#                 now_points = self.input_data[random_list[i+1]]
#                 target_np_noised = noise_Gaussian(now_points, 0.005)
#                 if (i == 0):
#                     plot_single_pcd(target_np_noised.squeeze(),'./res/partial_random.png') 
#                 now_points = (target_np_noised @ transform[:3, :3].T)+ transform[:3, 3]
#                 if (i == 0):
#                     plot_single_pcd(self.input_data[random_list[i+1]].squeeze(),'./res/partial_sys.png') 
                    
#                 transform = random_pose(60, 0.5)
#                 a = (now_points @ transform[:3, :3].T)+ transform[:3, 3]
#                 plot_single_pcd(a.squeeze(),'./res/partial'+str(i+1)+'.png') 
                
#             plot_single_pcd_res(self.input_data[random_list[1]].squeeze(),'./res/partial_error.png')
            
            
            
#             partial = np.concatenate((self.input_data[random_list[0]],self.input_data[random_list[1]],self.input_data[random_list[2]],self.input_data[random_list[3]]))
#             partial = torch.from_numpy(partial).unsqueeze(0)
#             plot_single_pcd(partial[0].detach().cpu().numpy().squeeze(),'./res/partial_all.png') 
#             print(partial.shape)
#             p_idx = furthest_point_sample(partial.contiguous(), 2048)   #0
#             pred_center = gather_points(partial.transpose(2,1).contiguous(), p_idx.int().contiguous()).permute(0,2,1) # 415061
# #             print(pred_center.shape)
# #             plot_single_pcd_res(pred_center.squeeze().cpu().numpy(),'./res/partial_res.png') 
# #             print(pred_center.shape)
#             complete = torch.from_numpy((self.gt_data[index // 26]))
#             plot_single_pcd_res(complete.squeeze(),'./res/complete.png') 
            
#             target_np_noised = noise_Gaussian(self.input_data[random_list[0]], 0.005)
            
#             b = self.input_data[random_list[0]]
#             transform = random_pose(5/ 180 * np.pi, 0.2)
#             c = self.input_data[random_list[1]]
#             d = self.input_data[random_list[3]]
#             c_ = (c @ transform[:3, :3].T)+ transform[:3, 3]
            
            
#             plot_single_pcd_all(b.squeeze(), c_.squeeze(), './res/regis_input.png')
            
#             plot_single_pcd_all(b.squeeze(), c.squeeze(), './res/regis_gt.png')
            
#             transform = random_pose(0.5/ 180 * np.pi, -0.005)
#             c_ = self.input_data[random_list[1]]
#             c_ = (c @ transform[:3, :3].T)+ transform[:3, 3]
#             plot_single_pcd_all(b.squeeze(), c_.squeeze(), './res/regis_sr-net.png')
            
# #             transform = random_pose(30/ 180 * np.pi, -0.1)
# #             c_ = self.input_data[random_list[13]]
# #             c_ = (c @ transform[:3, :3].T)+ transform[:3, 3]
#             odom_transform, _,_,src_scan_pts = icp(c, b, init_pose=None, max_iterations=20, gt = None)
#             plot_single_pcd_all(b.squeeze(), src_scan_pts[:3,:].squeeze().T, './res/regis_icp.png')
            
#             transform = random_pose(120/ 180 * np.pi, -0.1)
#             c_ = self.input_data[random_list[1]]
#             c_ = (c @ transform[:3, :3].T)+ transform[:3, 3]
#             plot_single_pcd_all(b.squeeze(), c_.squeeze(), './res/regis_idam.png')
            
#             odom_transform = pygicp.align_points(b, d)#, initial_guess = icp_initial)
#             m = c.shape[1]
#             src = np.ones((m+1,c.shape[0]))
#             src[:m,:] = np.copy(c.T)
#             src_gt_scan_pts = np.dot(odom_transform, src)[:3,:].transpose(1,0)
            
# #             transform = random_pose(180/ 180 * np.pi, -0.5)
# #             c_ = self.input_data[random_list[13]]
# #             c_ = (c @ transform[:3, :3].T)+ transform[:3, 3]
#             plot_single_pcd_all(b.squeeze(), src_gt_scan_pts.squeeze(), './res/regis_gmc.png')
#             print(dsadsa)
#             partial = torch.from_numpy(partial)
            
    
    
    
    
    
    
    
    
#             sys_error = np.random.normal(0, 0.001, (4, 3))
#             target_np_noised = noise_Gaussian(self.input_data[random_list[0]], 0.005)
#             partial = (target_np_noised + sys_error[0][np.newaxis, :])

#             for i in range(3):
#                 now_points = self.input_data[random_list[i+1]]
#                 target_np_noised = noise_Gaussian(now_points, 0.005)
#                 now_points = (target_np_noised + sys_error[i+1][np.newaxis, :])
#                 partial = np.concatenate((partial, now_points), 0)
                
#             partial = torch.from_numpy(partial)
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
            
#             plot_single_pcd(self.input_data[random_list[0]].squeeze(),'./res/ori.png')
            transform = random_pose(5/ 180 * np.pi, 0.2)
            c = self.input_data[random_list[1]]
            c_1 = (c @ transform[:3, :3].T)+ transform[:3, 3]
            
            plot_single_pcd_red(self.input_data[random_list[0]].squeeze(),'./res/a.png') 
            plot_single_pcd_blue(c_1.squeeze(),'./res/b.png')
            
#             plot_single_pcd(c_1.squeeze(),'./res/ori2.png')
#             transform = random_pose(0.02, 0.02)
#             c = self.input_data[random_list[2]]
#             c_2 = (c @ transform[:3, :3].T)+ transform[:3, 3]
#             plot_single_pcd(c_2.squeeze(),'./res/ori3.png')
#             transform = random_pose(0.03, 0)
#             c = self.input_data[random_list[3]]
#             c_3 = (c @ transform[:3, :3].T)+ transform[:3, 3]
#             plot_single_pcd(c_3.squeeze(),'./res/ori4.png')
            
#             plot_single_pcd(partial.detach().cpu().numpy().squeeze(),'./res/all.png')
#             plot_single_pcd2(partial.detach().cpu().numpy().squeeze(),'./res/sample.png')
#             plot_single_pcd3(partial.detach().cpu().numpy().squeeze(),'./res/sample_bad.png')
            
#             a1 = torch.from_numpy(self.input_data[random_list[0]]).unsqueeze(0)
#             a2 = torch.from_numpy(self.input_data[random_list[3]]).unsqueeze(0)
#             dist, idx = knn_point(8, a2, a1)
#             c_1 = index_points(torch.from_numpy(c_3).unsqueeze(0), idx)[:,:,0,:]
            
#             a1 = torch.from_numpy(self.input_data[random_list[0]]).unsqueeze(0)
#             a2 = torch.from_numpy(self.input_data[random_list[1]]).unsqueeze(0)
#             dist, idx = knn_point(8, a2, a1)
#             c_2 = index_points(torch.from_numpy(c_2).unsqueeze(0), idx)[:,:,0,:]
            plot_single_pcd_all(self.input_data[random_list[0]].squeeze(), self.input_data[random_list[1]].squeeze(), './res/ori.png')
            plot_single_pcd_all(self.input_data[random_list[0]].squeeze(), c_1.squeeze(), './res/registration.png')
#             plot_single_pcd_all2(self.input_data[random_list[0]].squeeze(), c_1.detach().cpu().numpy().squeeze(), './res/registration_bad.png')
#             plot_single_pcd_res(self.input_data[random_list[0]].squeeze(), self.input_data[random_list[1]].squeeze(), './res/registration_res.png')
            
            
#             pred_center = pred_center.detach().cpu().squeeze().numpy()
#             plot_single_pcd(pred_center.squeeze().detach().cpu().numpy().squeeze(),'./res/complete.png')
            
            
#             partial = np.concatenate([self.input_data[random_list[0]],c_1,c_2,c_3])
#             partial = torch.from_numpy(partial).contiguous().cuda().unsqueeze(0).float()
#             p_idx = furthest_point_sample(partial.contiguous(), 2048).cuda()  #0
#             print(partial.device)
#             print(p_idx.device)
#             pred_center = gather_points(partial.transpose(2,1).contiguous(), p_idx.int().contiguous()).permute(0,2,1)
#             print(pred_center.device)
#             pred_center = pred_center.squeeze().cpu().detach().numpy()
#             pred_center = noise_Gaussian(pred_center, 0.01)
#             plot_single_pcd(pred_center,'./res/complete_bad.png')
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