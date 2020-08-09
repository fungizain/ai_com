import os
import glob
import pathlib
import numpy as np
from PIL import Image, ImageDraw

import torch
from torch.utils.data import DataLoader
from torchvision import datasets

from facenet import InceptionResnetV1
from facenet import MTCNN, extract_face

def extract_img(model, img_path, save_path):
    img_name = img_path.name
    img = Image.open(str(img_path))
    face_tensor, prob = model(img, return_prob=True)
    boxes, probs, points = model.detect(img, landmarks=True)
    
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    
    if not save_path.exists():
        save_path.mkdir()
    
    if boxes is None or points is None:
        return False
    
    for i, (box, prob, point) in enumerate(zip(boxes, probs, points)):
        if prob < 0.97:
            continue
        draw.rectangle(box.tolist(), width=5)
        for p in point:
            draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=10)
        extract_face(img, box, save_path=save_path.joinpath(f'{img_name}_detected_face_{i}.jpg'))
    img_draw.save(save_path.joinpath(f'{img_name}_annotate_faces.jpg'))
    return True

def draw_img(model, img_dir, save_dir):
    images = []
    for suffix in ['jpg', 'jpeg', 'png']:
        images.extend(glob.glob(str(img_dir.joinpath(f'*/*.{suffix}'))))
    for img in images:
        img_path = pathlib.Path(img)
        save_path = save_dir.joinpath(img_path.parent.stem)
        extract_img(model, img_path, save_path)


if __name__ == "__main__":
    model = MTCNN(keep_all=True)
    resnet = InceptionResnetV1(pretrained='vggface2')
    
    data_dir = pathlib.Path('dataset/images/')
    test_dir = pathlib.Path('dataset/test/')
    draw_img(model, data_dir, test_dir)
    
    def collate_fn(x):
        return x[0]

    # dataset = datasets.ImageFolder(data_dir)
    # loader = DataLoader(dataset, collate_fn=collate_fn)