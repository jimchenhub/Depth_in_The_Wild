import matplotlib.pyplot as plt
import torch
import cv2
import h5py
import numpy as np


cmap = plt.cm.plasma
def colored_depthmap(depth, d_min=None, d_max=None):
    if d_min is None:
        d_min, _ = torch.min(depth, 1)
        d_min, _ = torch.min(d_min, 0)
    if d_max is None:
        d_max, _ = torch.max(depth, 1)
        d_max, _ = torch.max(d_max, 0)
    depth_relative = (depth - d_min) / (d_max - d_min)
    depth_relative = depth_relative.detach().cpu().numpy()
    colored_map = cmap(depth_relative)[:,:,:3] # H, W, C
    R = colored_map[:,:,0]
    G = colored_map[:,:,1]
    B = colored_map[:,:,2]
    new_colored_map = np.array([B, G, R])
    new_colored_map = np.transpose(new_colored_map, (1, 2, 0))
    return new_colored_map


if __name__ == '__main__':
    with h5py.File('C:/Users/jimch/Documents/NYU/data/654_NYU_MITpaper_test_imgs_orig_size/1_depth.h5') as f:
        depth_map = f["depth"][:]
    depth_map = torch.Tensor(depth_map)
    colored_map = colored_depthmap(depth_map)
    cv2.imshow('test', colored_map)
    cv2.waitKey(0)
    cv2.destroyWindow('test')
