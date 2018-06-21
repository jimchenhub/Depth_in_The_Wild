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
from config import PATH_PREFIX
from metrics import Metrics

g_input_width = 320
g_input_height = 240


def one_test(name, target, model, transform, checkpoint, device):
    # forward
    img = Image.open(name)
    # img = img.resize((g_input_width, g_input_height))
    data = transform(img)
    data = torch.unsqueeze(data, 0).to(device)
    output = model(data)
    output = torch.exp(output)
    # convert to depth map
    depth_map = torch.squeeze(output)
    # print(depth_map[100,100])
    # print(depth_map[100,400])
    d_min = torch.min(depth_map)
    colored_map = colored_depthmap(depth_map)

    # show result
    with h5py.File(target) as f:
        target = f["depth"][:]
    target = torch.Tensor(target)
    # target.resize_(g_input_height, g_input_width)
    target = target.to(device)
    # print(target[100,100])
    # print(target[100,400])
    target_map = colored_depthmap(target)
    result = np.hstack((target_map, colored_map))
    cv2.imshow('test', result)
    cv2.waitKey(0)
    cv2.destroyWindow('test')

    # metrics result
    # print(depth_map.type)
    return depth_map, target


def test(test_list, checkpoint, device):
    # model
    cudnn.benchmark = True
    transform = transforms.ToTensor()
    model = HourGlass()
    model = model.cuda()
    model_state = torch.load(checkpoint)['model_state']
    model.load_state_dict(model_state)
    # read list
    f = open(test_list, "r")
    content = f.read().split("\n")
    f.close()
    # for l in content:
    me = Metrics()
    result = np.zeros(6)
    for i in range(1, 601):
        # name = PATH_PREFIX+'Documents/NYU/data'+ l.split(" ")[0]
        # target = PATH_PREFIX+'Documents/NYU/data'+ l.split(" ")[1]
        name = PATH_PREFIX+'Documents/NYU/data/654_NYU_MITpaper_test_imgs_orig_size/'+str(i)+".png" 
        target = PATH_PREFIX+'Documents/NYU/data/654_NYU_MITpaper_test_imgs_orig_size/'+str(i)+"_depth.h5" 
        depth_map, target_map = one_test(name, target, model, transform, checkpoint, device)
        mse, rmse, mae, delta1, delta2, delta3 = me.single_metrics(depth_map.detach().cpu(), 
                                                                   target_map.detach().cpu())
        result[0] += mse
        result[1] += rmse
        result[2] += mae 
        result[3] += delta1
        result[4] += delta2
        result[5] += delta3
        if i % 50 == 0:
            print("finished ", str(i))
    result /= 600
    print("mse:       %f" % result[0])
    print("rmse:      %f" % result[1])
    print("mae:       %f" % result[2])
    print("<1.25:     %f" % result[3])
    print("<1.25^2:   %f" % result[4])
    print("<1.25^3:   %f" % result[5])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_list', type=str,
                        default=PATH_PREFIX+'Documents/NYU/data/test_list.txt')
    # parser.add_argument('--depth', type=str,
    #                     default=PATH_PREFIX+'Documents/NYU/data/654_NYU_MITpaper_test_imgs_orig_size/10_depth.h5')
    parser.add_argument('--checkpoint', default=PATH_PREFIX+"Documents/GitHub/Depth_in_The_Wild/results/2018-06-20-22-14-09/checkpoint_epoch_13.pth")
    args = parser.parse_args()
    # start training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    for i in range(19,20):
        print("checkpoint epoch: ", str(i))
        checkpoint = PATH_PREFIX+"Documents/GitHub/Depth_in_The_Wild/results/2018-06-21-16-44-52/checkpoint_epoch_"+str(i)+".pth"
        test(args.test_list, checkpoint, device)
