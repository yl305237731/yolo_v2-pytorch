import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from tools.util import to_cpu


class YoloLossLayer(nn.Module):
    def __init__(self, anchors, class_number, reduction, warmup_batches, coord_scale=5.0, noobj_scale=1,
                 obj_scale=5, class_scale=1.0, obj_thresh=0.5, net_factor=(416, 416), use_gpu=False, hard_conf=True):
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
        self.warmup_batches = warmup_batches
        self.use_gpu = use_gpu
        self.hard_conf = hard_conf
        self.warmup_keeper = 0
        self.net_factor = net_factor

    def forward(self, net_out, ground_truth):

        batch_size, grid_h, grid_w, out_channel = net_out.shape
        targets, anchor_mask, obj_mask = self.encode_target(ground_truth, grid_h, grid_w)
        conf_mask = anchor_mask.squeeze(dim=2)

        net_out = net_out.view(batch_size, self.anchor_number, self.class_number + 5, grid_h * grid_w)
        coords = torch.zeros_like(net_out[:, :, :4, :])

        if self.use_gpu:
            coords = coords.cuda()

        true_box_xy = targets[:, :, 0:2, :]
        true_box_wh = targets[:, :, 2:4, :]
        xywh_mask = anchor_mask

        coords[:, :, :2, :] = net_out[:, :, :2, :].sigmoid()
        coords[:, :, 2:4, :] = net_out[:, :, 2:4, :]
        conf = net_out[:, :, 4, :].sigmoid()
        clas = net_out[:, :, 5:, :].view(batch_size * self.anchor_number, self.class_number, grid_h * grid_w).transpose(1, 2).contiguous().view(-1, self.class_number)

        # classification loss
        clas_mask = anchor_mask.view(batch_size * self.anchor_number, 1, grid_h * grid_w).transpose(1, 2).contiguous().view(-1, 1)
        t_clas = targets[:, :, 5, :].view(batch_size * self.anchor_number, 1, grid_h * grid_w).transpose(1, 2).contiguous().view(-1, 1).long()
        clas_loss = F.cross_entropy(clas, t_clas.squeeze(dim=1), reduction='none') * clas_mask.squeeze(dim=1)

        # In the warm-up phase, let the grid without the target learn the aspect ratio of the anchor.
        if self.warmup_keeper < self.warmup_batches:
            self.warmup_batches += batch_size
            true_box_xy = (true_box_xy + 0.5) * (1 - anchor_mask) + true_box_xy
            xywh_mask = torch.ones_like(anchor_mask)

            if self.use_gpu:
                xywh_mask = xywh_mask.cuda()

        # coords loss
        anchor_w = self.anchors[:, 0].contiguous().view(self.anchor_number, 1)
        anchor_h = self.anchors[:, 1].contiguous().view(self.anchor_number, 1)

        if self.use_gpu:
            anchor_w = anchor_w.cuda()
            anchor_h = anchor_h.cuda()

        t_w_scale = true_box_wh[:, :, 0, :].exp() * anchor_w / self.net_factor[0]
        t_h_scale = true_box_wh[:, :, 1, :].exp() * anchor_h / self.net_factor[1]

        p_w_scale = coords[:, :, 2, :].exp() * anchor_w / self.net_factor[0]
        p_h_scale = coords[:, :, 3, :].exp() * anchor_h / self.net_factor[1]

        true_wh_scale = torch.cat([torch.unsqueeze(t_w_scale, 2), torch.unsqueeze(t_h_scale, 2)], dim=2)
        pre_wh_scale = torch.cat([torch.unsqueeze(p_w_scale, 2), torch.unsqueeze(p_h_scale, 2)], dim=2)

        xy_loss = F.mse_loss(coords[:, :, :2, :], true_box_xy, reduction='none') * xywh_mask
        wh_loss = F.mse_loss(pre_wh_scale, true_wh_scale, reduction='none') * xywh_mask
        coords_loss = xy_loss.sum() + wh_loss.sum()

        # confidence loss
        # 1. object confidence loss
        t_conf = targets[:, :, 4, :]
        obj_conf_loss = F.mse_loss(conf, t_conf, reduction='none') * conf_mask

        # 2. no-object confidence loss
        # 2.1 grid has no object
        noobj_conf_loss1 = F.mse_loss(conf, t_conf, reduction='none') * (1 - obj_mask)

        # 2.2 grid has object, but the anchor not responsible for object, if iou > obj_thresh, don't calculate loss,
        # else as background

        no_response_anchors_mask = (1 - conf_mask) * obj_mask
        grid_predicts = self.to_grid_coords(coords, grid_h, grid_w, anchor_w, anchor_h)
        grid_gts = self.to_grid_coords(targets[:, :, 0:4, :], grid_h, grid_w, anchor_w, anchor_h)

        iou_scores = self.compute_iou(grid_predicts, grid_gts)
        iou_gt_pred_mask = (iou_scores < self.obj_thresh).float()

        noobj_conf_loss2 = F.mse_loss(conf, t_conf, reduction='none') * no_response_anchors_mask * iou_gt_pred_mask

        noobj_conf_loss = noobj_conf_loss1 + noobj_conf_loss2
        total_loss = clas_loss.sum() * self.class_scale + coords_loss * self.coord_scale + \
                     obj_conf_loss.sum() * self.obj_scale + noobj_conf_loss.sum() * self.noobj_scale

        # Compute some online statistics
        count = conf_mask.sum()
        count_noobj = (1 - obj_mask).sum()
        recall_50 = ((iou_scores >= 0.5).float() * conf_mask).sum() / (count + 1e-3)
        recall_75 = ((iou_scores >= 0.75).float() * conf_mask).sum() / (count + 1e-3)
        avg_iou = (iou_scores * conf_mask).sum() / (count + 1e-3)
        avg_obj = (conf * conf_mask).sum() / (count + 1e-3)
        avg_noobj = (conf * (1 - obj_mask)).sum() / (count_noobj + 1e-3)
        obj_conf = list(zip(*torch.where(conf * conf_mask)))

        for i in range(len(obj_conf)):
            print(" obj {}  confidence: {}, prids coords: {}, t_coords: {}".format(i, conf[obj_conf[i]],
                  to_cpu(grid_predicts[obj_conf[i][0], obj_conf[i][1], :, obj_conf[i][2]]),
                  to_cpu(grid_gts[obj_conf[i][0], obj_conf[i][1], :, obj_conf[i][2]])))
        print(" avg_obj_conf: {}\n avg_noobj_cof: {}\n avg_iou: {}\n count obj: {}\n count noobj: {}\n iou > 50 recall:"
              "{}\n iou >75 recall: {}".format(avg_obj, avg_noobj, avg_iou, count, count_noobj, recall_50, recall_75))

        print("total loss: {}, class loss: {}, coords loss: {}, obj_conf loss: {}, noobj_conf loss: {}".format(
            total_loss, clas_loss.sum() * self.class_scale, coords_loss * self.coord_scale, obj_conf_loss.sum() *
            self.obj_scale, noobj_conf_loss.sum() * self.noobj_scale))
        print("--------------------------------------------------------")

        return total_loss

    def to_grid_coords(self, coords, grid_h, grid_w, anchor_w, anchor_h):
        col_index = torch.arange(0, grid_w).repeat(grid_h, 1).view(grid_h * grid_w)
        row_index = torch.arange(0, grid_h).repeat(grid_w, 1).t().contiguous().view(grid_h * grid_h)
        grid_boxs = torch.zeros_like(coords)
        if self.use_gpu:
            col_index = col_index.cuda()
            row_index = row_index.cuda()
            grid_boxs = grid_boxs.cuda()
        # to grid size
        grid_boxs[:, :, 0, :] = (coords[:, :, 0, :] + col_index.float()) / grid_w * self.net_factor[0]
        grid_boxs[:, :, 1, :] = (coords[:, :, 1, :] + row_index.float()) / grid_h * self.net_factor[1]
        grid_boxs[:, :, 2, :] = coords[:, :, 2, :].exp() * anchor_w
        grid_boxs[:, :, 3, :] = coords[:, :, 3, :].exp() * anchor_h

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

                print("obj best iou: {}, best anchor index: {}".format(best_iou, best_anchor_index))
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
                # Whether to use IOU as the gt of onf, it is relatively more difficult to train with IOU
                if self.hard_conf:
                    obj_conf = best_iou
                else:
                    obj_conf = torch.Tensor([1])
                grid_info = torch.cat([x_offset.view(-1, 1), y_offset.view(-1, 1), w_log.view(-1, 1), h_log.view(-1, 1), obj_conf.view(-1, 1), cls.view(-1, 1)], dim=1)
                targets[b, best_anchor_index, :, row * grid_w + col] = grid_info.clone()
                anchor_mask[b, best_anchor_index, :, row * grid_w + col] = 1
                obj_mask[b, :, row * grid_w + col] = 1

        return targets, anchor_mask, obj_mask

    def anchor_iou(self, box, anchor):
        """
        Move the box and anchor box to the top left corner to overlap, calculate IOU
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
            return area_inter / (area_boxa + area_boxb - area_inter + 1e-5)

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
