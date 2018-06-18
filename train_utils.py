from shutil import copyfile
import torch
from torch.utils.data import DataLoader


def prep_img(img, device):
    return torch.Torch(img.unsqueeze(0)).to(device)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def _fit_epoch(model, loader, criterion, optimizer, device):
    model.train()
    loss_meter = AverageMeter()
    t = iter(loader)
    count = 0
    for data, target in t:
        data = data.to(device)
        target['x_A'] = target['x_A'].to(device)
        target['y_A'] = target['y_A'].to(device)
        target['x_B'] = target['x_B'].to(device)
        target['y_B'] = target['y_B'].to(device)
        target['ordinal_relation'] = target['ordinal_relation'].to(device)
        output = model(data)
        loss = criterion(output, target)
        loss_meter.update(loss.item())
        if count % 10 == 0:
            print("[ iteration {:d}, loss: {:.6f} ]".format(count, loss_meter.avg))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        count += 1

    return loss_meter.avg


def fit(model, train_data, criterion, optimizer, scheduler, device, batch_size=32,
        shuffle=True, nb_epoch=1, validation_data=None, num_workers=4):
    if validation_data:
        print('Train on {} samples, Validate on {} samples'.format(len(train_data), len(validation_data)))
    else:
        print('Train on {} samples'.format(len(train_data)))

    train_loader = DataLoader(train_data, batch_size, shuffle, num_workers=num_workers, pin_memory=True)
    for i in range(nb_epoch):
        print("epoch ", str(i+1))
        scheduler.step()
        print("learning rate: ", optimizer.param_groups[0]['lr'])
        _fit_epoch(model, train_loader, criterion, optimizer, device)


def validate(model, validation_data, criterion, batch_size, device):
    model.eval()
    val_loss = AverageMeter()
    loader = DataLoader(validation_data, batch_size=batch_size, shuffle=True)
    for data, target in loader:
        data = torch.Tensor(data, device=device)
        target['x_A'] = target['x_A'].to(device)
        target['y_A'] = target['y_A'].to(device)
        target['x_B'] = target['x_B'].to(device)
        target['y_B'] = target['y_B'].to(device)
        target['ordinal_relation'] = target['ordinal_relation'].to(device)
        output = model(data)
        loss = criterion(output, target)
        val_loss.update(loss.data[0])
    return val_loss.avg


def save_checkpoint(model_state, optimizer_state, filename, epoch=None, is_best=False):
    state = dict(model_state=model_state,
                 optimizer_state=optimizer_state,
                 epoch=epoch)
    torch.save(state, filename)
    if is_best:
        copyfile(filename, 'model_best.pth.tar')
