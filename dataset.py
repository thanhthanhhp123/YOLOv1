from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor

def get_dataset(batch_size = 64):
    transform = Compose([
        Resize((448, 448)),
        ToTensor()
    ])

    train_dataset = datasets.VOCDetection(root = 'data', year = '2007', image_set = 'train', download = True, transform = transform)
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True)

    return train_loader


import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
import xml.etree.ElementTree as ET
import numpy as np

# Định nghĩa danh sách các lớp đối tượng trong VOC
VOC_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow",
    "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

class VOCDataset(data.Dataset):
    def __init__(self, root, image_set='trainval', S=7, B=2, C=20, transform=None):
        """
        Args:
            root (string): Root directory of the VOC2007 dataset.
            image_set (string): 'train', 'val', 'trainval', 'test'.
            S (int): Grid size.
            B (int): Number of bounding boxes per grid cell.
            C (int): Number of classes.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root = root
        self.image_set = image_set
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform

        # Đường dẫn đến các thư mục cần thiết
        self.image_dir = os.path.join(self.root, 'JPEGImages')
        self.annotation_dir = os.path.join(self.root, 'Annotations')
        self.image_set_dir = os.path.join(self.root, 'ImageSets', 'Main')

        # Đọc danh sách các file hình ảnh
        with open(os.path.join(self.image_set_dir, f'{image_set}.txt'), 'r') as f:
            self.image_ids = f.read().strip().split()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        # Lấy ID của hình ảnh
        image_id = self.image_ids[index]

        # Đọc hình ảnh
        img_path = os.path.join(self.image_dir, f'{image_id}.jpg')
        image = Image.open(img_path).convert("RGB")

        # Đọc annotation
        annotation_path = os.path.join(self.annotation_dir, f'{image_id}.xml')
        boxes, labels = self.parse_voc_xml(annotation_path)

        # Áp dụng các phép biến đổi
        if self.transform:
            image, boxes = self.transform(image, boxes)

        # Chuyển đổi bounding boxes thành định dạng YOLOv1
        target = self.convert_to_yolo_format(boxes, labels)

        return image, target

    def parse_voc_xml(self, annotation_path):
        """
        Parse a VOC XML file and return bounding boxes and labels.
        """
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.iter('object'):
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue  # Bỏ qua các đối tượng khó

            name = obj.find('name').text.lower().strip()
            if name not in VOC_CLASSES:
                continue  # Bỏ qua các lớp không trong danh sách

            label = VOC_CLASSES.index(name)

            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        return boxes, labels

    def convert_to_yolo_format(self, boxes, labels):
        """
        Convert bounding boxes to YOLOv1 format: SxSx(B*5 + C)
        """
        grid_size = self.S
        cell_size = 1.0 / grid_size
        target = torch.zeros((grid_size, grid_size, self.B * 5 + self.C))

        for i in range(len(boxes)):
            box = boxes[i]
            label = labels[i]

            # Tính toán trung tâm và kích thước của bounding box
            x_center = (box[0] + box[2]) / 2.0
            y_center = (box[1] + box[3]) / 2.0
            width = box[2] - box[0]
            height = box[3] - box[1]

            # Chuẩn hóa tọa độ
            x_center /= Image.width
            y_center /= Image.height
            width /= Image.width
            height /= Image.height

            # Xác định ô lưới chứa trung tâm bounding box
            i_cell = int(x_center * grid_size)
            j_cell = int(y_center * grid_size)

            # Tính toán vị trí trong ô lưới
            x_cell = x_center * grid_size - i_cell
            y_cell = y_center * grid_size - j_cell

            # Chọn bounding box có độ tin cậy thấp nhất để gán
            # Để đơn giản, ta gán vào bounding box đầu tiên nếu chưa có
            if target[j_cell, i_cell, 4] == 0:
                box_idx = 0
            elif target[j_cell, i_cell, 9] == 0:
                box_idx = 1
            else:
                continue  # Nếu cả hai bounding boxes đã được gán, bỏ qua

            # Gán thông tin bounding box
            target[j_cell, i_cell, box_idx * 5 + 0] = x_cell
            target[j_cell, i_cell, box_idx * 5 + 1] = y_cell
            target[j_cell, i_cell, box_idx * 5 + 2] = np.sqrt(width)
            target[j_cell, i_cell, box_idx * 5 + 3] = np.sqrt(height)
            target[j_cell, i_cell, box_idx * 5 + 4] = 1.0  # Độ tin cậy

            # Gán nhãn lớp (sử dụng one-hot encoding)
            target[j_cell, i_cell, self.B * 5 + label] = 1.0

        return target

# Lưu ý: Trong hàm convert_to_yolo_format, bạn cần truy cập vào kích thước hình ảnh.
# Vì vậy, chúng ta cần sửa đổi một chút để truyền kích thước hình ảnh vào hàm này.
# Sửa đổi class VOCDataset như sau:

class VOCDataset(data.Dataset):
    def __init__(self, root, image_set='trainval', S=7, B=2, C=20, transform=None):
        self.root = root
        self.image_set = image_set
        self.S = S
        self.B = B
        self.C = C
        self.transform = transform

        self.image_dir = os.path.join(self.root, 'JPEGImages')
        self.annotation_dir = os.path.join(self.root, 'Annotations')
        self.image_set_dir = os.path.join(self.root, 'ImageSets', 'Main')

        with open(os.path.join(self.image_set_dir, f'{image_set}.txt'), 'r') as f:
            self.image_ids = f.read().strip().split()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, index):
        image_id = self.image_ids[index]

        img_path = os.path.join(self.image_dir, f'{image_id}.jpg')
        image = Image.open(img_path).convert("RGB")
        img_width, img_height = image.size

        annotation_path = os.path.join(self.annotation_dir, f'{image_id}.xml')
        boxes, labels = self.parse_voc_xml(annotation_path)

        if self.transform:
            image, boxes = self.transform(image, boxes)

        target = self.convert_to_yolo_format(boxes, labels, img_width, img_height)

        return image, target

    def parse_voc_xml(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        boxes = []
        labels = []

        for obj in root.iter('object'):
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue

            name = obj.find('name').text.lower().strip()
            if name not in VOC_CLASSES:
                continue

            label = VOC_CLASSES.index(name)

            bndbox = obj.find('bndbox')
            xmin = float(bndbox.find('xmin').text)
            ymin = float(bndbox.find('ymin').text)
            xmax = float(bndbox.find('xmax').text)
            ymax = float(bndbox.find('ymax').text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(label)

        return boxes, labels

    def convert_to_yolo_format(self, boxes, labels, img_width, img_height):
        grid_size = self.S
        target = torch.zeros((grid_size, grid_size, self.B * 5 + self.C))

        for i in range(len(boxes)):
            box = boxes[i]
            label = labels[i]

            # Tính toán trung tâm và kích thước của bounding box
            x_center = (box[0] + box[2]) / 2.0
            y_center = (box[1] + box[3]) / 2.0
            width = box[2] - box[0]
            height = box[3] - box[1]

            # Chuẩn hóa tọa độ
            x_center_norm = x_center / img_width
            y_center_norm = y_center / img_height
            width_norm = width / img_width
            height_norm = height / img_height

            # Xác định ô lưới chứa trung tâm bounding box
            i_cell = int(x_center_norm * grid_size)
            j_cell = int(y_center_norm * grid_size)

            # Đảm bảo i_cell, j_cell không vượt quá giới hạn
            i_cell = min(i_cell, grid_size - 1)
            j_cell = min(j_cell, grid_size - 1)

            # Tính toán vị trí trong ô lưới
            x_cell = x_center_norm * grid_size - i_cell
            y_cell = y_center_norm * grid_size - j_cell

            # Chọn bounding box có độ tin cậy thấp nhất để gán
            # Để đơn giản, ta gán vào bounding box đầu tiên nếu chưa có
            if target[j_cell, i_cell, 4] == 0:
                box_idx = 0
            elif target[j_cell, i_cell, 9] == 0:
                box_idx = 1
            else:
                continue  # Nếu cả hai bounding boxes đã được gán, bỏ qua

            # Gán thông tin bounding box
            target[j_cell, i_cell, box_idx * 5 + 0] = x_cell
            target[j_cell, i_cell, box_idx * 5 + 1] = y_cell
            target[j_cell, i_cell, box_idx * 5 + 2] = np.sqrt(width_norm)
            target[j_cell, i_cell, box_idx * 5 + 3] = np.sqrt(height_norm)
            target[j_cell, i_cell, box_idx * 5 + 4] = 1.0  # Độ tin cậy

            # Gán nhãn lớp (sử dụng one-hot encoding)
            target[j_cell, i_cell, self.B * 5 + label] = 1.0

        return target