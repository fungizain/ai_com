import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader


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

#-----------------------------------------------------------------------------------

train_txt_path = Path("./dataset/afg-pt")

class TensorFolder(Dataset):
    def __init__(self, pt_dir):
        super(TensorFolder, self).__init__()
        
        classes = [d.name for d in pt_dir.iterdir() if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.pts = list(pt_dir.glob('*/*.pt'))

    def __getitem__(self, index):
        pt = self.pts[index]
        ft = torch.load(pt)
        label = self.class_to_idx[pt.parent.stem]

        return ft, label

    def __len__(self):
        return len(self.pts)

if __name__ == "__main__":
    batch_size = 256
    data_dir = Path("./dataset/afg-pt")

    # train_ds = TensorFolder(data_dir.joinpath('train'))
    valid_ds = TensorFolder(data_dir.joinpath('val'))
    valid_dl = DataLoader(valid_ds, batch_size*2, num_workers=0)
