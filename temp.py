import os
import glob
import random
import shutil
from tqdm import tqdm
from pathlib import Path

def list_dirs(dir):
    return [f for f in dir.iterdir() if f.is_dir()]

def list_files(dir):
    return [f for f in dir.iterdir() if f.is_file() and not f.name.startswith(".")]

def setup_files(class_dir, seed):
    random.seed(seed)
    files = list_files(class_dir)
    files.sort()
    random.shuffle(files)
    return files

def split_files(files, split_train_idx, split_val_idx, use_test):
    files_train = files[:split_train_idx]
    files_val = (
        files[split_train_idx:split_val_idx] if use_test else files[split_train_idx:]
    )
    
    li = [(files_train, "train"), (files_val, "val")]
    
    if use_test:
        files_test = files[split_val_idx:]
        li.append((files_test, "test"))
    return li

def copy_files(files_type, class_dir, output, prog_bar):
    class_name = class_dir.name
    for (files, folder_type) in files_type:
        full_path = Path(output, folder_type, class_name)
        full_path.mkdir(parents=True, exist_ok=True)
        for f in files:
            if not prog_bar is None:
                prog_bar.update()
            if type(f) == tuple:
                for x in f:
                    shutil.copy2(x, full_path)
            else:
                shutil.copy2(f, full_path)

def split_class_dir_ratio(class_dir, output, ratio, seed, prog_bar):
    files = setup_files(class_dir, seed)
    
    split_train_idx = int(ratio[0] * len(files))
    split_val_idx = split_train_idx + int(ratio[1] * len(files))
    
    li = split_files(files, split_train_idx, split_val_idx, len(ratio) == 3)
    copy_files(li, class_dir, output, prog_bar)
    
def ratio(input, output="output", seed=420, ratio=(0.8, 0.1, 0.1)):
    assert round(sum(ratio), 5) == 1
    assert len(ratio) in (2, 3)
    
    prog_bar = tqdm(desc="Copying files", unit=" files")
    
    for class_dir in list_dirs(input):
        split_class_dir_ratio(class_dir, output, ratio, seed, prog_bar)
    
    prog_bar.close()

# ----------------------------------------------------------------------------
data_dir = Path("../dataset/AFAD-Lite-Gender/")

am_dir = Path(data_dir, 'Male/')
af_dir = Path(data_dir, 'Female/')

def ageGender_to_gender():
    am_img = glob.glob('../dataset/AFAD-Lite/*/111/*.jpg')
    af_img = glob.glob('../dataset/AFAD-Lite/*/112/*.jpg')
    am_dir.mkdir(exist_ok=True)
    af_dir.mkdir(exist_ok=True)
    
    for path in tqdm(am_img):
        path = Path(path)
        os.rename(path, Path(am_dir, path.name))
    for path in tqdm(af_img):
        path = Path(path)
        os.rename(path, Path(af_dir, path.name))
        
        
if __name__ == "__main__":
    ratio(data_dir)