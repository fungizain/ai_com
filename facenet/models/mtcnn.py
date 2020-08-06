import os
import torch
import torch.nn as nn
import numpy as np

from .utils.detect_face import detect_face, extract_face

class PNet(nn.Module):
    """
    Pretrained MTCNN PNet.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 10, kernel_size=3)
        self.prelu1 = nn.PReLU(10)
        self.pool1 = nn.MaxPool2d(2, 2, ceil_mode=True)
        
        self.conv2 = nn.Conv2d(10, 16, kernel_size=3)
        self.prelu2 = nn.PReLU(16)
        
        self.conv3 = nn.Conv2d(16, 32, kernel_size=3)
        self.prelu3 = nn.PReLU(32)
        
        self.conv4_1 = nn.Conv2d(32, 2, kernel_size=1)
        self.softmax4_1 = nn.Softmax(dim=1)
        self.conv4_2 = nn.Conv2d(32, 4, kernel_size=1)
        
        self.training = False
        
        if pretrained:
            state_dict_path = os.path.join(os.path.dirname(__file__), '../data/pnet.pt')
            state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        
        a = self.conv4_1(x)
        a = self.softmax4_1(a)
        
        b = self.conv4_2(x)
        
        return b, a
            
class RNet(nn.Module):
    """
    Pretrained MTCNN RNet.
    """
    def __init__(self, pretrained=True):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 28, kernel_size=3)
        self.prelu1 = nn.PReLU(28)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        
        self.conv2 = nn.Conv2d(28, 48, kernel_size=3)
        self.prelu2 = nn.PReLU(48)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        
        self.conv3 = nn.Conv2d(48, 64, kernel_size=2)
        self.prelu3 = nn.PReLU(64)
        
        self.dense4 = nn.Linear(576, 128)
        self.prelu4 = nn.PReLU(128)
        
        self.dense5_1 = nn.Linear(128, 2)
        self.softmax5_1 = nn.Softmax(dim=1)
        self.dense5_2 = nn.Linear(128, 4)
        
        self.training = False
        
        if pretrained:
            state_dict_path = os.path.join(os.path.dirname(__file__), '../data/rnet.pt')
            state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)
            
    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense4(x.view(x.shape[0], -1))
        x = self.prelu4(x)
        
        a = self.dense5_1(x)
        a = self.softmax5_1(a)
        
        b = self.dense5_2(x)
        
        return b, a
            
class ONet(nn.Module):
    """
    Pretrained MTCNN ONet.
    """
    def __init__(self, pretrained=True):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.prelu1 = nn.PReLU(32)
        self.pool1 = nn.MaxPool2d(3, 2, ceil_mode=True)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.prelu2 = nn.PReLU(64)
        self.pool2 = nn.MaxPool2d(3, 2, ceil_mode=True)
        
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3)
        self.prelu3 = nn.PReLU(64)
        self.pool3 = nn.MaxPool2d(2, 2, ceil_mode=True)
        
        self.conv4 = nn.Conv2d(64, 128, kernel_size=2)
        self.prelu4 = nn.PReLU(128)
        
        self.dense5 = nn.Linear(1152, 256)
        self.prelu5 = nn.PReLU(256)
        
        self.dense6_1 = nn.Linear(256, 2)
        self.softmax6_1 = nn.Softmax(dim=1)
        self.dense6_2 = nn.Linear(256, 4)
        self.dense6_3 = nn.Linear(256, 10)

        self.training = False

        if pretrained:
            state_dict_path = os.path.join(os.path.dirname(__file__), '../data/onet.pt')
            state_dict = torch.load(state_dict_path)
            self.load_state_dict(state_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.prelu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.prelu4(x)
        
        x = x.permute(0, 3, 2, 1).contiguous()
        x = self.dense5(x.view(x.shape[0], -1))
        x = self.prelu5(x)
        
        a = self.dense6_1(x)
        a = self.softmax6_1(a)
        b = self.dense6_2(x)
        c = self.dense6_3(x)
        
        return b, c, a
    
class MTCNN(nn.Module):
    """
    MTCNN face detection module.
    Returns images cropped to include the face only.
    Input:
        - PIL image or list of PIL images
        - numpy.ndarray (uint8) representing either a single image or a batch of images.
    """
    def __init__(
            self, image_size=160, margin=0, min_face_size=20,
            thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
            select_largest=True, keep_all=False, device=None
            ):
        super().__init__()
        
        self.image_size = image_size
        self.margin = margin
        self.min_face_size = min_face_size
        self.thresholds = thresholds
        self.factor = factor
        self.post_process = post_process
        self.select_largest = select_largest
        self.keep_all = keep_all
        
        self.pnet = PNet()
        self.rnet = RNet()
        self.onet = ONet()
        
        self.device = torch.device('cpu')
        if device is not None:
            self.device = device
            self.to(device)
            
    def forward(self, img, save_path=None, return_prob=False):
        """
        Example:
        >>> from facenet import MTCNN
        >>> mtcnn = MTCNN()
        >>> face_tensor, prob = mtcnn(img, save_path='face.png', return_prob=True)
        """
        with torch.no_grad():
            batch_boxes, batch_probs = self.detect(img)
            
        batch_mode = True
        if not isinstance(img, (list, tuple)) and not (isinstance(img, np.ndarray) and len(img.shape) == 4):
            img = [img]
            batch_boxes = [batch_boxes]
            batch_probs = [batch_probs]
            batch_mode = False

        # Parse save path(s)
        if save_path is not None:
            if isinstance(save_path, str):
                save_path = [save_path]
        else:
            save_path = [None for _ in range(len(img))]
        
        # Process all bounding boxes and probabilities
        faces, probs = [], []
        for im, box_im, prob_im, path_im in zip(img, batch_boxes, batch_probs, save_path):
            if box_im is None:
                faces.append(None)
                probs.append([None] if self.keep_all else None)
                continue

            if not self.keep_all:
                box_im = box_im[[0]]

            faces_im = []
            for i, box in enumerate(box_im):
                face_path = path_im
                if path_im is not None and i > 0:
                    save_name, ext = os.path.splitext(path_im)
                    face_path = save_name + '_' + str(i + 1) + ext

                face = extract_face(im, box, self.image_size, self.margin, face_path, self.device)
                if self.post_process:
                    face = (face - 127.5) * 0.0078125
                faces_im.append(face)

            if self.keep_all:
                faces_im = torch.stack(faces_im)
            else:
                faces_im = faces_im[0]
                prob_im = prob_im[0]
            
            faces.append(faces_im)
            probs.append(prob_im)
    
        if not batch_mode:
            faces = faces[0]
            probs = probs[0]

        if return_prob:
            return faces, probs
        else:
            return faces
        
    def detect(self, img, landmarks=False):
        """
        Example:
        >>> from PIL import Image, ImageDraw
        >>> from facenet import MTCNN, extract_face
        >>> mtcnn = MTCNN(keep_all=True)
        >>> boxes, probs, points = mtcnn.detect(img, landmarks=True)
        >>> img_draw = img.copy()
        >>> draw = ImageDraw.Draw(img_draw)
        >>> for i, (box, point) in enumerate(zip(boxes, points)):
        ...     draw.rectangle(box.tolist(), width=5)
        ...     for p in point:
        ...         draw.rectangle((p - 10).tolist() + (p + 10).tolist(), width=10)
        ...     extract_face(img, box, save_path=f'detected_face_{i}.png')
        >>> img_draw.save('annotated_faces.png')
        """
        with torch.no_grad():
            batch_boxes, batch_points = detect_face(
                img, self.min_face_size,
                self.pnet, self.rnet, self.onet,
                self.thresholds, self.factor, self.device
                )
            
        boxes, probs, points = [], [], []
        for box, point in zip(batch_boxes, batch_points):
            box = np.array(box)
            point = np.array(point)
            if len(box) == 0:
                boxes.append(None)
                probs.append([None])
                points.append(None)
            elif self.select_largest:
                box_order = np.argsort((box[:, 2] - box[:, 0]) * (box[:, 3] - box[:, 1]))[::-1]
                box = box[box_order]
                point = point[box_order]
                boxes.append(box[:, :4])
                probs.append(box[:, 4])
                points.append(point)
            else:
                boxes.append(box[:, :4])
                probs.append(box[:, 4])
                points.append(point)
        boxes = np.array(boxes)
        probs = np.array(probs)
        points = np.array(points)
        
        if not isinstance(img, (list, tuple)) and not (isinstance(img, np.ndarray) and len(img.shape) == 4):
            boxes = boxes[0]
            probs = probs[0]
            points = points[0]

        if landmarks:
            return boxes, probs, points

        return boxes, probs