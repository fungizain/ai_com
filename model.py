import os
import time
import glob
import torch
import PIL
from PIL import Image, ImageDraw

from facenet import InceptionResnetV1, MTCNN

def input_tolist(input_):
    if not isinstance(input_, (list, tuple)):
        input_ = [input_]
    return input_

def convert(input_):
    if isinstance(input_, str):
        input_ = Image.open(input_)
    if isinstance(input_, PIL.Image.Image):
        return input_
    else:
       print('Input type shd be \"path(str)\" or \"PIL.Image.Image\" or \"numpy.ndarray\".')

def cal_face_tensor(cnn, input_, prob_threshold=0.98):
    input_ = convert(input_)
    fts, probs = cnn(input_, return_prob=True)
    if fts is not None:
        if fts.dim() == 3:
            fts = fts.unsqueeze(0)
        else:
            fts = [ft.unsqueeze(0) for ft, prob in zip(fts, probs) if prob>prob_threshold]
            fts = torch.cat(fts)
    else:
        fts = torch.ones(1, 3, 160, 160).to(cnn.device)
    return fts

def cal_embs_dist(embs_1, embs_2, dist_intial=100):
    r1 = embs_1.shape[0]
    c2 = embs_2.shape[0]

    dist = torch.ones(r1, c2) * dist_intial

    for i in range(r1):
        for j in range(c2):
            emb1 = embs_1[i].unsqueeze(0)
            emb2 = embs_2[j].unsqueeze(0)
            dist[i][j] = (emb1 - emb2).norm().item()

    return dist

def mt_overlap(dist):
    mean, std = dist.mean(), dist.std()
    margin = mean - std
    row_mins = dist.argmin(dim=1)
    col_mins = dist.argmin(dim=0)

    a = set()
    b = set()
    for i, j in enumerate(row_mins):
        if dist[i][j] < margin:
            a.add((i, j))
    for j, i in enumerate(col_mins):
        if dist[i][j] < margin:
            b.add((i, j))

    overlap = []
    for _ in a and b:
        overlap.append(_[1])
    return overlap

def img_draw_with_box(img, box):
    img_draw = img.copy()
    draw = ImageDraw.Draw(img_draw)
    draw.rectangle(box.tolist(), width=5)
    return img_draw

class AI_Model():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.cnn = MTCNN(device=self.device, keep_all=True)
        self.cnn0 = MTCNN(device=self.device)
        self.rnn1 = InceptionResnetV1(device=self.device, pretrained='vggface2').eval()
        # self.rnn1 = InceptionResnetV1(device=self.device, pretrained='casia-webface').eval()

    def compare(self, input_1, input_2, type_=None, show_imgs=False, save_path='test/'):
        ind_dic = {}

        if type_ == "S001":
            cnn1 = self.cnn0
            cnn2 = self.cnn0
            rnn = self.rnn1
            dist = self.compare_(cnn1, cnn2, rnn, input_1, input_2)
            dist = dist.mean(dim=0)
            ans_index = dist.argmin().item()

        elif type_ == "S002":
            cnn1 = self.cnn
            cnn2 = self.cnn0
            rnn = self.rnn1
            dist = self.compare_(cnn1, cnn2, rnn, input_1, input_2)
            dist = dist.argmin(dim=0)
            ans_index = dist.mode().values.item()

        elif type_ == "S003":
            cnn1 = self.cnn0
            cnn2 = self.cnn0
            rnn = self.rnn1
            dist = self.compare_(cnn1, cnn2, rnn, input_1, input_2)
            ans_index = dist.argmin().item()
        
        elif type_ == "S004":
            cnn = self.cnn0
            rnn = self.rnn1
            dist = self.detect_gender(cnn, rnn, input_1)
            ans_index = 0

        elif type_ == "M001":
            cnn1 = self.cnn
            cnn2 = self.cnn0
            rnn = self.rnn1
            dist = self.compare_(cnn1, cnn2, rnn, input_1, input_2)
            ans_index = mt_overlap(dist)

        elif type_ == "D001":
            cnn1 = self.cnn0
            cnn2 = self.cnn
            rnn = self.rnn1
            dist = self.compare_(cnn1, cnn2, rnn, input_1, input_2)
            ind = dist.argmin().item()
            ans_index = self.show_eyes_xy(cnn2, input_2, ind)
        
        else:
            print("Please select correct \"type\".")
            return False
        
        if show_imgs:
            self.show_img(cnn1, cnn2, input_1, input_2, ind_dic, save_path)

        return ans_index

    def compare_(self, cnn1, cnn2, rnn, input_1, input_2):
        input_1_list = input_tolist(input_1)
        input_2_list = input_tolist(input_2)

        fts_1, fts_2 = [], []
        for input_1 in input_1_list:
            ft_1 = cal_face_tensor(cnn1, input_1)
            fts_1.append(ft_1)

        for input_2 in input_2_list:
            ft_2 = cal_face_tensor(cnn2, input_2)
            fts_2.append(ft_2)

        fts_1 = torch.cat(fts_1)
        fts_2 = torch.cat(fts_2)

        embs_1 = rnn(fts_1)
        embs_2 = rnn(fts_2)

        dist = cal_embs_dist(embs_1, embs_2)
        return dist

    def detect_gender(self, cnn, rnn, input_):
        # input_ = convert(input_)
        # fts = cnn(input_)
        # embs = rnn(fts)
        pass

    def show_eyes_xy(self, cnn, input, ind):
        input = convert(input)
        _, _, points = cnn.detect(input, True)

        return points[ind][:2].tolist()


if __name__ == "__main__":
    start = time.time()
    
    # in_1 = './dataset/images/gidle.jpeg'
    # in_1 = glob.glob('./dataset/images/miyeon/*.jpg')[3]
    in_1 = './dataset/images/gidle2.jpeg'
    in_2 = glob.glob('./dataset/images/*.jpg')

    model = AI_Model()
    ans = model.compare(in_1, in_2, type_="M001")
    
    time_used = round(time.time() - start, 2)
    print(f'time_used={time_used}s')
    
    print([in_2[_] for _ in ans])