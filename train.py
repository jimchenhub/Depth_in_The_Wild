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


def main(data_path, label_path, nb_epoch, save_path, device, start_path=None,
         batch_size=2, lr=1e-3, plot_history=True):
    cudnn.benchmark = True
    train_data = NYUDepth(data_path, label_path, transforms=transforms.ToTensor())
    hourglass = HourGlass()
    hourglass = hourglass.cuda()
    optimizer = RMSprop(hourglass.parameters(), lr)
    scheduler = MultiStepLR(optimizer, milestones=[3, 6], gamma=0.1)

    if start_path:
        experiment = torch.load(start_path)
        hourglass.load_state_dict(experiment['model_state'])
        optimizer.load_state_dict(experiment['optimizer_state'])
    criterion = RelativeDepthLoss()

    history = fit(hourglass, train_data, criterion, optimizer, scheduler, device,
                  batch_size=batch_size, nb_epoch=nb_epoch)
    save_checkpoint(hourglass.state_dict(), optimizer.state_dict(), save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default=PATH_PREFIX+'Documents/NYU/data/795_NYU_MITpaper_train_imgs')
    parser.add_argument('--label_path', type=str, default=PATH_PREFIX+'Documents/NYU/data/labels_train.pkl')
    parser.add_argument('--nb_epoch', default=3, type=int, help='Epochs')
    parser.add_argument('--save_path', default=PATH_PREFIX+"Documents/GitHub/Depth_in_The_Wild/results/test_result.pth")
    parser.add_argument('--start_path', default=None)
    parser.add_argument('--batch_size', default=4)
    parser.add_argument('--lr', default=1e-3)
    args = parser.parse_args()
    # start training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(args.data_path, args.label_path, args.nb_epoch, args.save_path, device,
         args.start_path, args.batch_size, args.lr)
