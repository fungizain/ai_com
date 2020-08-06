import os
import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import ImageFolder
import torchvision.transforms as tt
from torch.utils.data import DataLoader

from .facenet.models.resnet import build_resnet


def cal_mean_std(train_ds, batch_size=10, num_workers=3):
    dl = DataLoader(train_ds, batch_size=batch_size, num_workers=num_workers)
    mean, std = 0., 0.
    for img, _ in dl:
        img = img.view(batch_size, 3, -1)
        mean += img.mean(2).sum(0)
        std += img.std(2).sum(0)
    mean /= len(train_ds)
    std /= len(train_ds)
    if type(mean) == torch.Tensor:
        mean = [round(_, 4) for _ in mean.tolist()]
        std = [round(_, 4) for _ in std.tolist()]
    return mean, std

def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

class DeviceDataLoader():
    def __init__(self, dl):
        self.dl = dl
        self.device = self.get_default_device()

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)
    
    @staticmethod
    def get_default_device():
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    
def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

@torch.no_grad()
def evaluate(model, val_loader):
    model.eval()
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_cycle(epochs, max_lr, model, train_loader, val_loader,
    weight_decay=0, grad_clip=None, opt_func=torch.optim.SGD):
    torch.cuda.empty_cache()
    history = []
    train_log = []

    optimizer = opt_func(model.parameters(), max_lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr, epochs=epochs,
                                                steps_per_epoch=len(train_loader))
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_losses = []
        lrs = []
        
        pbar = tqdm(train_loader)
        pbar.set_description(f"{epoch=}, v_acc=     ")
        for batch in train_loader:
            loss = model.training_step(batch)
            train_losses.append(loss)
            loss.backward()

            if grad_clip:
                nn.utils.clip_grad_value_(model.parameters(), grad_clip)
                
            optimizer.step()
            optimizer.zero_grad()
            
            lrs.append(get_lr(optimizer))
            sched.step()
            pbar.update(1)
            
        result = evaluate(model, val_loader)
        result['train_loss'] = torch.stack(train_losses).mean().item()
        result['lrs'] = lrs
        train_log.append(model.epoch_end(epoch, result))
        history.append(result)
        v_acc = round(100 * result['val_acc'], 2)
        pbar.set_description(f"{epoch=}, {v_acc=}")
        pbar.close()
    return train_log, history

def plot_accuracies(history, dir):
    accuracies = [x['val_acc'] for x in history]
    plt.plot(accuracies, '-x')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy vs. No. of epochs')
    plt.savefig(dir + '/accuracies.png')
    plt.close()

def plot_losses(history, dir):
    train_losses = [x.get('train_loss') for x in history]
    val_losses = [x['val_loss'] for x in history]
    plt.plot(train_losses, '-bx')
    plt.plot(val_losses, '-rx')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Training', 'Validation'])
    plt.title('Loss vs. No. of epochs')
    plt.savefig(dir + '/losses.png')
    plt.close()

def plot_lrs(history, dir):
    lrs = np.concatenate([x.get('lrs', []) for x in history])
    plt.plot(lrs)
    plt.xlabel('Batch no.')
    plt.ylabel('Learning rate')
    plt.title('Learning Rate vs. Batch no.')
    plt.savefig(dir + '/learning_rates.png')
    plt.close()

def grid_search(train_dl, valid_dl, epochs, opt_func, max_lrs, grad_clips, weight_decays):
    today = date.today().strftime("%d-%m-%Y")
    model_dir = f'model/{today}'
    device = DeviceDataLoader.get_default_device()
    if not os.path.exists(model_dir): os.makedirs(model_dir)
    for max_lr, grad_clip, weight_decay in list(itertools.product(max_lrs, grad_clips, weight_decays)):
        print(f'{epochs=}, {max_lr=}, {grad_clip=}, {weight_decay=}')
        model = to_device(build_resnet('resnet18', 'fanin', 10, False), device)
        history = [evaluate(model, valid_dl)]
        start = time.time()
        train_log, history_ = fit_one_cycle(epochs, max_lr, model, train_dl, valid_dl,
                                            grad_clip=grad_clip, weight_decay=weight_decay, opt_func=opt_func)
        history += history_
        time_used = round(time.time() - start, 4)
        score = round(history[-1]['val_acc'] * 100, 4)
        
        if score >= 90:
            folder = model_dir + '/acc_{:.2f}'.format(score)
            if not os.path.exists(folder): os.makedirs(folder)
            with open(folder + '/params.txt', 'w') as f:
                f.write(f'{max_lr=}\n')
                f.write(f'{grad_clip=}\n')
                f.write(f'{weight_decay=}\n')
                f.write(f'{time_used=}\n\n')
                f.write('log:')
                [f.write(_) for _ in train_log]
            plot_accuracies(history, folder)
            plot_losses(history, folder)
            plot_lrs(history, folder)
            torch.save(model.state_dict(), folder 
                       + '/cifar10-resnet9.pth')
        print(f"\n{score=}, {time_used=}s\n")

        del model, train_log, history_, history
        torch.cuda.empty_cache()
        
if __name__ == "__main__":
    batch_size = 100
    data_dir = "./data/cifar10"
    train_ds = ImageFolder(data_dir + '/train', tt.ToTensor())
    valid_ds = ImageFolder(data_dir + '/test', tt.ToTensor())
    # stats = cal_mean_std(train_ds)
    # print(stats)
    stats = ([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.201])
    train_tf = tt.Compose([tt.RandomCrop(32, padding=4, padding_mode='reflect'),
                            tt.RandomHorizontalFlip(),
                            tt.ToTensor(),
                            tt.Normalize(*stats, inplace=True)])
    valid_tf = tt.Compose([tt.ToTensor(),
                            tt.Normalize(*stats)])
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0, pin_memory=True)
    valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=0, pin_memory=True)
    train_dl = DeviceDataLoader(train_dl)
    valid_dl = DeviceDataLoader(valid_dl)
    
    epochs = 10
    max_lrs = [5e-3]
    grad_clips = [1e-1]
    weight_decays = [1e-3]
    opt_func = torch.optim.Adam
    
    grid_search(train_dl, valid_dl, epochs, opt_func, max_lrs, grad_clips, weight_decays)