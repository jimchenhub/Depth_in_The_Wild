import os
import datetime
import argparse
from torchvision import transforms
from torch.optim import RMSprop
from torch.optim.lr_scheduler import MultiStepLR
import torch

from datasets import NYUDepth
from models import HourGlass
from criterion import RelativeDepthLoss
from train_utils import fit, save_checkpoint
from torch.backends import cudnn
from config import PATH_PREFIX


def main(train_data_path, train_label_path, val_data_path, val_label_path, 
         nb_epoch, save_path, device, start_path, batch_size, lr):
    cudnn.benchmark = True
    train_data = NYUDepth(train_data_path, train_label_path, transforms=transforms.ToTensor())
    val_data = NYUDepth(val_data_path, val_label_path, transforms=transforms.ToTensor())
    hourglass = HourGlass()
    hourglass = hourglass.cuda()
    optimizer = RMSprop(hourglass.parameters(), lr, weight_decay=1e-5)
    # scheduler = MultiStepLR(optimizer, milestones=[10, 20], gamma=0.1)
    scheduler = None

    if start_path:
        experiment = torch.load(start_path)
        hourglass.load_state_dict(experiment['model_state'])
        optimizer.load_state_dict(experiment['optimizer_state'])
    criterion = RelativeDepthLoss()

    # save path
    t_now = datetime.datetime.now()
    t = t_now.strftime("%Y-%m-%d-%H-%M-%S")
    save_path = os.path.join(save_path, t)
    if not os.path.isdir(save_path):
        os.mkdir(save_path)

    history = fit(hourglass, train_data, val_data, criterion, optimizer, scheduler, 
                  save_path, device, batch_size=batch_size, nb_epoch=nb_epoch)
    # save final model
    save_checkpoint(hourglass.state_dict(), optimizer.state_dict(), os.path.join(save_path, "test_result.pth"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_data_path', type=str, 
                        default=PATH_PREFIX+'Documents/NYU/data/795_NYU_MITpaper_train_imgs')
    parser.add_argument('--train_label_path', type=str, 
                        default=PATH_PREFIX+'Documents/NYU/data/labels_train.pkl')
    parser.add_argument('--val_data_path', type=str, 
                        default=PATH_PREFIX+'Documents/NYU/data/795_NYU_MITpaper_train_imgs')
    parser.add_argument('--val_label_path', type=str, 
                        default=PATH_PREFIX+'Documents/NYU/data/labels_val.pkl')
    parser.add_argument('--nb_epoch', default=50, type=int, help='Epochs')
    parser.add_argument('--save_path', default=PATH_PREFIX+"Documents/GitHub/Depth_in_The_Wild/results/")
    parser.add_argument('--start_path', default=None)
    parser.add_argument('--batch_size', default=4)
    parser.add_argument('--lr', default=1e-3)
    args = parser.parse_args()
    # start training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(args.train_data_path, args.train_label_path, args.val_data_path, args.val_label_path, 
         args.nb_epoch, args.save_path, device, args.start_path, args.batch_size, args.lr)
