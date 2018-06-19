import argparse
from PIL import Image
import cv2
import h5py
import numpy as np
from torchvision import transforms
import torch

from models import HourGlass
from torch.backends import cudnn
from visual import colored_depthmap
from train_utils import load_checkpoint
from config import PATH_PREFIX


def test(name, target, checkpoint, device):
    cudnn.benchmark = True
    transform=transforms.ToTensor()
    model = HourGlass()
    model = model.cuda()
    model_state = load_checkpoint(checkpoint)
    model.load_state_dict(model_state)
    # forward
    img = Image.open(name)
    data = transform(img)
    data = torch.unsqueeze(data, 0).to(device)
    output = model(data)
    # convert to depth map
    depth_map = torch.squeeze(output)
    colored_map = colored_depthmap(depth_map)

    # show result
    with h5py.File(target) as f:
        target = f["depth"][:]
    target = torch.Tensor(target)
    target_map = colored_depthmap(target)
    print(target_map.shape)
    print(colored_map.shape)
    result = np.hstack((target_map, colored_map))
    cv2.imshow('test', result)
    cv2.waitKey(0)
    cv2.destroyWindow('test')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=PATH_PREFIX+'Documents/NYU/data/654_NYU_MITpaper_test_imgs_orig_size/1.png')
    parser.add_argument('--depth', type=str, default=PATH_PREFIX+'Documents/NYU/data/654_NYU_MITpaper_test_imgs_orig_size/1_depth.h5')
    parser.add_argument('--checkpoint', default=PATH_PREFIX+"Documents/GitHub/Depth_in_The_Wild/results/test_result.pth")
    args = parser.parse_args()
    # start training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test(args.data, args.depth, args.checkpoint, device)
