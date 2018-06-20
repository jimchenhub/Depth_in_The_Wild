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
from metrics import Metrics


def one_test(name, target, model, transform, checkpoint, device):
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
    target = target.to(device)
    target_map = colored_depthmap(target)
    # result = np.hstack((target_map, colored_map))
    # cv2.imshow('test', result)
    # cv2.waitKey(0)
    # cv2.destroyWindow('test')

    # metrics result
    # print(depth_map.type)
    me = Metrics()
    me.show_metrics([depth_map.detach().cpu()], [target.detach().cpu()])


def test(test_list, checkpoint, device):
    # model
    cudnn.benchmark = True
    transform=transforms.ToTensor()
    model = HourGlass()
    model = model.cuda()
    model_state = load_checkpoint(checkpoint)
    model.load_state_dict(model_state)
    # read list
    f = open(test_list, "r")
    content = f.read().split("\n")
    f.close()
    for l in content:
        name = l.split(" ")[0]
        target = l.split(" ")[1]
        one_test(name, target, model, transform, checkpoint, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_list', type=str,
                        default=PATH_PREFIX+'Documents/NYU/data/test_list.txt')
    # parser.add_argument('--depth', type=str,
    #                     default=PATH_PREFIX+'Documents/NYU/data/654_NYU_MITpaper_test_imgs_orig_size/10_depth.h5')
    parser.add_argument('--checkpoint', default=PATH_PREFIX+"Documents/GitHub/Depth_in_The_Wild/results/test_result.pth")
    args = parser.parse_args()
    # start training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    test(args.test_list, args.checkpoint, device)
