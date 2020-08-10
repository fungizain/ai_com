import os

from train_utils import to_device, DeviceDataLoader

if __name__ == "__main__":
    batch_size = 250
    data_dir = Path("./dataset/afg")

    stats = ([0.5553, 0.4449, 0.3953], [0.2345, 0.2058, 0.1853])
    train_tf = tt.Compose([tt.Resize(160),
                           tt.RandomCrop(32, padding=4, padding_mode='reflect'),
                           tt.RandomHorizontalFlip(),
                           tt.ToTensor(),
                           tt.Normalize(*stats, inplace=True)])
    valid_tf = tt.Compose([tt.Resize(160),
                           tt.ToTensor(),
                           tt.Normalize(*stats)])
    
    train_ds = ImageFolder(data_dir.joinpath('train/'), transform=train_tf)
    valid_ds = ImageFolder(data_dir.joinpath('val/'), transform=valid_tf)
    
    train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=0)
    valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=0)
    train_dl = DeviceDataLoader(train_dl)
    valid_dl = DeviceDataLoader(valid_dl)
    
    model_name = 'se-resnext50-32x4d'

    build_resnet('se-resnext50-32x4d', 'classic', 2, False)