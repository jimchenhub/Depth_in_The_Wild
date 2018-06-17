import argparse
from torchvision import transforms
from torch.optim import RMSprop
import torch
import matplotlib.pyplot as plt

from datasets import NYUDepth
from models import HourGlass
from criterion import RelativeDepthLoss
from train_utils import fit, save_checkpoint
from torch.backends import cudnn


def main(data_path, label_path, nb_epoch, save_path, start_path=None,
         batch_size=2, lr=1e-3, plot_history=True, device=None):
    cudnn.benchmark = True
    train_data = NYUDepth(data_path, label_path, transforms=transforms.ToTensor())
    hourglass = HourGlass()
    hourglass = hourglass.cuda()
    optimizer = RMSprop(hourglass.parameters(), lr)

    if start_path:
        experiment = torch.load(start_path)
        hourglass.load_state_dict(experiment['model_state'])
        optimizer.load_state_dict(experiment['optimizer_state'])
    criterion = RelativeDepthLoss()

    history = fit(hourglass, train_data, criterion, optimizer, batch_size, nb_epoch, device)
    save_checkpoint(hourglass.state_dict(), optimizer.state_dict(), save_path)
    if plot_history:
        plt.plot(history['loss'], label='loss')
        plt.xlabel('epoch')
        plt.ylabel('relative depth loss')
        plt.legend()
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='C:/Users/jimch/Documents/NYU/data/795_NYU_MITpaper_train_imgs')
    parser.add_argument('--label_path', type=str, default='C:/Users/jimch/Documents/NYU/data/labels_train.pkl')
    parser.add_argument('--nb_epoch', default=10, type=int, help='Epochs')
    parser.add_argument('--save_path', default=None)
    parser.add_argument('--start_path', default=None)
    parser.add_argument('--batch_size', default=4)
    parser.add_argument('--lr', default=1e-3)
    args = parser.parse_args()
    # start training
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    main(args.data_path, args.label_path, args.nb_epoch, args.save_path,
         args.start_path, args.batch_size, args.lr, device)
