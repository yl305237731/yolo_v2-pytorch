import numpy as np
import cv2
import torchvision.transforms as transforms
import torch
import os
import xml.etree.ElementTree as ET
from torch.utils.data.dataloader import default_collate


def adjust_learning_rate(initial_lr, optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    warmup_epoch = -1
    if epoch <= warmup_epoch:
        lr = 1e-6 + (initial_lr-1e-6) * iteration / (epoch_size * warmup_epoch)
    else:
        lr = initial_lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def custom_collate_fn(batch):
    items = list(zip(*batch))
    items[0] = default_collate(items[0])
    items[1] = list(items[1])
    return items


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c = c
        self.classes = classes
        self.label = np.argmax(self.classes)
        self.score = -1

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.label >= 0 if self.label else 0]

        return self.score


def decode_netout(netout, anchors, confidence_thresh, net_h=416, net_w=416):
    grid_h, grid_w = netout.shape[:2]
    nb_box = len(anchors)
    netout = netout.view(nb_box, -1, grid_w * grid_h)
    netout_mask = (netout[:, 4, :].sigmoid_() >= confidence_thresh).float()
    obj_grids = list(zip(*torch.where(netout_mask == 1)))
    boxes = []

    for ii in obj_grids:
        anchor_index = ii[0]
        grid_index = ii[1]
        box = netout[anchor_index, :, grid_index]
        row = grid_index // grid_h
        col = grid_index % grid_w
        x, y, w, h = box[0].sigmoid(), box[1].sigmoid(), box[2], box[3]
        anchor_w = anchors[anchor_index][0]
        anchor_h = anchors[anchor_index][1]
        x = (x + col) * (net_w / grid_w)
        y = (y + row) * (net_h / grid_h)
        w = anchor_w * np.exp(w)
        h = anchor_h * np.exp(h)
        with torch.no_grad():
            classes = torch.nn.functional.softmax(box[5:], 0)
        box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, box[4], classes)
        boxes.append(box)
    return boxes


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):

    for i in range(len(boxes)):

        boxes[i].xmin = int(boxes[i].xmin * image_w / net_w)
        boxes[i].xmax = int(boxes[i].xmax * image_w / net_w)
        boxes[i].ymin = int(boxes[i].ymin * image_h / net_h)
        boxes[i].ymax = int(boxes[i].ymax * image_h / net_h)


def iou(box_a, box_b):
    """
    :param box_a: [x1, y1, x2, y2]
    :param box_b: [x1, y1, x2, y2]
    :return: iou
    """
    area_boxa = (box_a.xmax - box_a.xmin) * (box_a.ymax - box_a.ymin)
    area_boxb = (box_b.xmax - box_b.xmin) * (box_b.ymax - box_b.ymin)

    def intersection(box1, box2):
        x_lt = max(box1.xmin, box2.xmin)
        y_lt = max(box1.ymin, box2.ymin)
        x_br = min(box1.xmax, box2.xmax)
        y_br = min(box1.ymax, box2.ymax)
        inter_w = max(x_br - x_lt, 0)
        inter_h = max(y_br - y_lt, 0)
        return float(inter_w * inter_h)
    area_inter = intersection(box_a, box_b)
    return area_inter / (area_boxa + area_boxb - area_inter)


def do_nms(boxes, nms_thresh=0.4):
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return

    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0
                    boxes[index_j].label = -1


def preprocess_input(image, net_w, net_h):
    resized = cv2.resize(image, (net_w, net_h))
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    return torch.unsqueeze(transform(resized), dim=0)


def draw_boxes(image, boxes, labels):
    for box in boxes:
        label = box.label
        if label >= 0:
            label_str = str(labels[label] + ',' + str(box.c.numpy()))
            cv2.rectangle(img=image, pt1=(box.xmin, box.ymin), pt2=(box.xmax, box.ymax), color=(255, 0, 255), thickness=2)
            cv2.putText(img=image,
                        text=label_str,
                        org=(box.xmin + 13, box.ymin - 13),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1e-3 * image.shape[0],
                        color=(255, 0, 0),
                        thickness=2)
    return image


def parse_voc_annotation(ann_dir, img_dir, labels=[]):
    all_insts = []
    for ann in sorted(os.listdir(ann_dir)):
        img = {'object': []}
        try:
            tree = ET.parse(os.path.join(ann_dir, ann))
        except Exception as e:
            print('Ignore this bad annotation: ' + ann_dir + ann)
            continue

        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = img_dir + elem.text
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}

                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text
                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]

                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            all_insts += [img]
    return all_insts


def to_cpu(x):
    return x.detach().cpu()
