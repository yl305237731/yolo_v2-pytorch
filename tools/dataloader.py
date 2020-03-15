import cv2
import os
import torch
import xml.etree.ElementTree as ET
import random
from torch.utils.data import Dataset


class VOCDataSet(Dataset):
    def __init__(self, img_dir, xml_dir, name_list, target_size=(416, 416), shuffle=True, augmentation=None, transform=None):
        """
        :param img_dir: images root dir
        :param xml_dir: xml file root dir
        :param name_list: like ['cat', 'dog',...]
        :param target_size: target image size: tuple(w, h)
        :param shuffle: data shuffle
        :param augmentation: data augmentation
        :param transform: torch transform
        """
        self.img_dir = img_dir
        self.xml_dir = xml_dir
        self.target_size = target_size
        self.name_list = name_list
        self.transform = transform
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.img_names, self.img_bboxs = self.parse_xml()

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        """

        :param idx:
        :return: img [3, h, w], img_bbox [x1, y1, x2, y2, class_index]
        """
        img_name = self.img_names[idx]
        img_bbox = self.img_bboxs[idx]
        img_ori = cv2.imread(os.path.join(self.img_dir, img_name))
        img = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
        if self.augmentation:
            img, img_bbox = self.augmentation.augment(img, img_bbox, noise=False, withName=True)
        height, width, _ = img.shape
        img = cv2.resize(img, self.target_size)
        img_bbox = torch.floor(torch.Tensor(img_bbox) * torch.Tensor([self.target_size[0] / width, self.target_size[1] / height,
                                                                      self.target_size[0] / width, self.target_size[1] / height, 1]))
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.tensor(img).permute([2, 0, 1])
        return img, img_bbox

    def get_label_index(self, name):
        return self.name_list.index(name)

    def parse_xml(self):
        img_names = []
        img_bboxs = []
        xml_dir = os.listdir(self.xml_dir)

        if self.shuffle:
            random.shuffle(xml_dir)

        for xml_name in xml_dir:
            xml_path = os.path.join(self.xml_dir, xml_name)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            img_name = tree.find('filename').text
            if not os.path.exists(os.path.join(self.img_dir, img_name)):
                continue
            img_names.append(img_name)
            objs = root.findall('object')
            box_info = list()
            for ix, obj in enumerate(objs):
                name = obj.find('name').text
                if name in self.name_list:
                    box = obj.find('bndbox')
                    x_min = int(float(box.find('xmin').text))
                    y_min = int(float(box.find('ymin').text))
                    x_max = int(float(box.find('xmax').text))
                    y_max = int(float(box.find('ymax').text))
                    label_index = self.get_label_index(name)
                    box_info.append([x_min, y_min, x_max, y_max, label_index])
            if len(box_info) <= 0:
                img_names.remove(img_name)
            else:
                img_bboxs.append(box_info)
        return img_names, img_bboxs

    def imshow(self, img, bboxs, widname=' '):
        for idx, bbox in enumerate(bboxs):
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 1)
        cv2.imshow(widname, img)
        cv2.waitKey(0)
