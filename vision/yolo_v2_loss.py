import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class YoloLossLayer(nn.Module):
    def __init__(self, anchors, class_number, reduction, coord_scale=5.0, noobj_scale=1,
                 obj_scale=5.0, class_scale=1.0, obj_thresh=0.6, use_gpu=False):
        super(YoloLossLayer, self).__init__()
        self.anchor_number = len(anchors)
        self.anchors = torch.Tensor(anchors)
        self.class_number = class_number
        self.reduction = reduction
        self.coord_scale = coord_scale
        self.noobj_scale = noobj_scale
        self.obj_scale = obj_scale
        self.class_scale = class_scale
        self.obj_thresh = obj_thresh
        self.use_gpu = use_gpu

    def forward(self, net_out, ground_truth):

        batch_size, grid_h, grid_w, out_channel = net_out.shape
        targets, anchor_mask, obj_mask = self.encode_target(ground_truth, grid_h, grid_w)
        conf_mask = anchor_mask.squeeze(dim=2)

        net_out = net_out.view(batch_size, self.anchor_number, self.class_number + 5, grid_h * grid_w)
        coords = torch.zeros_like(net_out[:, :, :4, :])

        if self.use_gpu:
            coords = coords.cuda()
        coords[:, :, :2, :] = net_out[:, :, :2, :].sigmoid()
        coords[:, :, 2:4, :] = net_out[:, :, 2:4, :]
        conf = net_out[:, :, 4, :].sigmoid()
        clas = net_out[:, :, 5:, :].view(batch_size * self.anchor_number, self.class_number, grid_h * grid_w).transpose(1, 2).contiguous().view(-1, self.class_number)

        # classification loss
        clas_mask = anchor_mask.view(batch_size * self.anchor_number, 1, grid_h * grid_w).transpose(1, 2).contiguous().view(-1, 1)
        t_clas = targets[:, :, 5, :].view(batch_size * self.anchor_number, 1, grid_h * grid_w).transpose(1, 2).contiguous().view(-1, 1).long()
        clas_loss = F.cross_entropy(clas, t_clas.squeeze(dim=1), reduction='none') * clas_mask.squeeze(dim=1)

        # coords loss
        t_coords = targets[:, :, :4, :]
        coords_loss = F.mse_loss(coords, t_coords, reduction='none') * anchor_mask

        # confidence loss
        # 1. object confidence loss
        t_conf = targets[:, :, 4, :]
        obj_conf_loss = F.mse_loss(conf, t_conf, reduction='none') * conf_mask

        # 2. no-object confidence loss
        # 2.1 grid has no object
        noobj_conf_loss1 = F.mse_loss(conf, t_conf, reduction='none') * (1 - obj_mask)

        # 2.2 grid has object, but the anchor not response to object, if iou > 0.6, dont calculate loss, else as background

        no_response_anchors_mask = (1 - conf_mask) * obj_mask
        grid_predicts = self.to_grid_coords(coords, grid_h, grid_w)
        grid_gts = self.to_grid_coords(t_coords, grid_h, grid_w)
        iou_gt_pred_mask = (self.compute_iou(grid_predicts, grid_gts) < self.obj_thresh).float()
        noobj_conf_loss2 = F.mse_loss(conf, t_conf, reduction='none') * no_response_anchors_mask * iou_gt_pred_mask

        noobj_conf_loss = noobj_conf_loss1 + noobj_conf_loss2
        total_loss = clas_loss.mean() * self.class_scale + coords_loss.mean() * self.coord_scale + \
                     obj_conf_loss.mean() * self.obj_scale + noobj_conf_loss.mean() * self.noobj_scale
        print("total loss: {}, class loss: {}, coords loss: {}, obj_conf loss: {}, noobj_conf loss: {}".format(
            total_loss, clas_loss.sum(), coords_loss.sum(), obj_conf_loss.sum(), noobj_conf_loss.sum()))
        return total_loss

    def to_grid_coords(self, coords, grid_h, grid_w):
        col_index = torch.arange(0, grid_w).repeat(grid_h, 1).view(grid_h * grid_w)
        row_index = torch.arange(0, grid_h).repeat(grid_w, 1).t().contiguous().view(grid_h * grid_h)
        anchor_w = self.anchors[:, 0].contiguous().view(self.anchor_number, 1)
        anchor_h = self.anchors[:, 1].contiguous().view(self.anchor_number, 1)
        grid_boxs = torch.zeros_like(coords)
        if self.use_gpu:
            grid_boxs.cuda()
        # to grid size
        grid_boxs[:, :, 0, :] = coords[:, :, 0, :] + col_index.float()
        grid_boxs[:, :, 1, :] = coords[:, :, 1, :] + row_index.float()
        grid_boxs[:, :, 2, :] = coords[:, :, 2, :].exp() * anchor_w / self.reduction
        grid_boxs[:, :, 3, :] = coords[:, :, 3, :].exp() * anchor_h / self.reduction

        # to [x1, y1, x2, y2]
        grid_boxs[:, :, 0, :] = grid_boxs[:, :, 0, :] - grid_boxs[:, :, 2, :] / 2
        grid_boxs[:, :, 1, :] = grid_boxs[:, :, 1, :] - grid_boxs[:, :, 3, :] / 2
        grid_boxs[:, :, 2, :] = grid_boxs[:, :, 0, :] + grid_boxs[:, :, 2, :]
        grid_boxs[:, :, 3, :] = grid_boxs[:, :, 1, :] + grid_boxs[:, :, 3, :]
        return grid_boxs

    def encode_target(self, ground_truth, grid_h, grid_w):
        batch_size = len(ground_truth)
        anchor_mask = torch.zeros(batch_size, self.anchor_number, 1, grid_h * grid_w)
        obj_mask = torch.zeros(batch_size, self.anchor_number, grid_h * grid_w)
        targets = torch.zeros(batch_size, self.anchor_number, 5 + 1, grid_h * grid_w)
        if self.use_gpu:
            anchor_mask = anchor_mask.cuda()
            obj_mask = obj_mask.cuda()
            targets = targets.cuda()

        for b in range(batch_size):
            bboxs = ground_truth[b]
            if len(bboxs) == 0:
                continue
            for box in bboxs:
                best_iou = 0
                best_anchor_index = 0
                cls = box[4]
                for ii, anchor in enumerate(self.anchors):
                    cur_iou = self.anchor_iou(box, anchor)
                    if cur_iou > best_iou:
                        best_iou = cur_iou
                        best_anchor_index = ii

                anchor_w = self.anchors[best_anchor_index][0]
                anchor_h = self.anchors[best_anchor_index][1]
                w = box[2] - box[0]
                h = box[3] - box[1]
                xc = box[0] + w / 2
                yc = box[1] + h / 2
                col = math.floor(xc / self.reduction)
                row = math.floor(yc / self.reduction)
                x_offset = xc / self.reduction - col
                y_offset = yc / self.reduction - row
                w_log = torch.log(w / anchor_w)
                h_log = torch.log(h / anchor_h)
                grid_info = torch.cat([x_offset.view(-1, 1), y_offset.view(-1, 1), w_log.view(-1, 1), h_log.view(-1, 1), best_iou.view(-1, 1), cls.view(-1, 1)], dim=1)
                targets[b, best_anchor_index, :, row * grid_w + col] = grid_info.clone()
                anchor_mask[b, best_anchor_index, :, row * grid_w + col] = 1
                obj_mask[b, :, row * grid_w + col] = 1

        return targets, anchor_mask, obj_mask

    def anchor_iou(self, box, anchor):
        """
        将box与anchor box 移到左上角重叠, 计算IOU
        :param box:
        :param anchor:
        :return:
        """

        def single_iou(box_a, box_b):
            area_boxa = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
            area_boxb = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

            def intersection(box1, box2):
                x_lt = max(box1[0], box2[0])
                y_lt = max(box1[1], box2[1])
                x_br = min(box1[2], box2[2])
                y_br = min(box1[3], box2[3])
                inter_w = max(x_br - x_lt, 0)
                inter_h = max(y_br - y_lt, 0)
                return float(inter_w * inter_h)

            area_inter = intersection(box_a, box_b)
            return area_inter / (area_boxa + area_boxb - area_inter)

        box = [0, 0, box[2] - box[0], box[3] - box[1]]
        anchor = [0, 0, anchor[0], anchor[1]]
        return single_iou(box, anchor)

    def compute_iou(self, boxes1, boxes2):
        tl = torch.max(boxes1[:, :, :2, :], boxes2[:, :, :2, :])
        br = torch.min(boxes1[:, :, 2:, :], boxes2[:, :, 2:, :])
        wh = br - tl
        wh = torch.max(wh, torch.zeros_like(wh))
        inter = wh[:, :, 0, :] * wh[:, :, 1, :]
        area_1 = (boxes1[:, :, 2, :] - boxes1[:, :, 0, :]) * (boxes1[:, :, 3, :] - boxes1[:, :, 1, :])
        area_2 = (boxes2[:, :, 2, :] - boxes2[:, :, 0, :]) * (boxes2[:, :, 3, :] - boxes2[:, :, 1, :])

        return inter / (area_1 + area_2 - inter + 1e-5)
