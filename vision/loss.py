import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class YoloLossLayer(nn.Module):
    def __init__(self, anchors, class_number, reduction, coord_scale=5.0, noobj_scale=1,
                 obj_scale=5, class_scale=1.0, obj_thresh=0.5, net_factor=(416, 416), max_box_per_image=30, use_gpu=False, hard_conf=True):
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
        self.hard_conf = hard_conf
        self.max_box_per_image = max_box_per_image
        self.net_factor = net_factor

    def forward(self, net_out, ground_truth):
        # 网络输出shape
        batch_size, grid_h, grid_w, out_channel = net_out.shape
        targets, anchor_mask, true_bboxs = self.encode_target(ground_truth, grid_h, grid_w)
        anchor_mask = anchor_mask.unsqueeze(dim=2)

        # 将网络的输出变形, 方便统一计算
        net_out = net_out.view(batch_size, self.anchor_number, self.class_number + 5, grid_h * grid_w)
        coords = torch.zeros_like(net_out[:, :, :4, :])

        if self.use_gpu:
            coords = coords.cuda()
        # 解码网络输出
        # 中心坐标偏移量
        coords[:, :, :2, :] = net_out[:, :, :2, :].sigmoid()
        # 宽和高
        coords[:, :, 2:4, :] = net_out[:, :, 2:4, :]
        # 置信度
        p_conf = net_out[:, :, 4, :].sigmoid()
        # 类别预测, 并view 成多分类交叉熵函数需要的格式
        clas = net_out[:, :, 5:, :].view(batch_size * self.anchor_number, self.class_number, grid_h * grid_w).transpose(
            1, 2).contiguous().view(-1, self.class_number)

        # anchor mask view 成 BCE 输出格式
        clas_mask = anchor_mask.view(batch_size * self.anchor_number, 1, grid_h * grid_w).transpose(1, 2).contiguous().view(-1, 1)
        # class gt view 成 BCE 输入格式
        t_clas = targets[:, :, 5, :].view(batch_size * self.anchor_number, 1, grid_h * grid_w).transpose(1, 2).contiguous().view(-1, 1).long()
        # 计算 class mask 中激活的anchor的分类损失
        clas_loss = F.cross_entropy(clas, t_clas.squeeze(dim=1), reduction='none') * clas_mask.squeeze(dim=1)

        # coords loss
        wh_loss_scale = 2.0 - 1.0 * targets[:, :, 2:3, :] * targets[:, :, 3:4, :] / (self.net_factor[0] * self.net_factor[1])
        xy_loss = F.mse_loss(coords[:, :, :2, :], targets[:, :, 0:2, :], reduction='none') * anchor_mask * wh_loss_scale
        wh_loss = F.mse_loss(coords[:, :, 2:4, :], targets[:, :, 2:4, :], reduction='none') * anchor_mask * wh_loss_scale
        coords_loss = xy_loss.sum() + wh_loss.sum()

        # 置信度损失
        t_conf = targets[:, :, 4, :]
        grid_predicts = self.rescale_to_img(coords, grid_h, grid_w).permute(0, 3, 1, 2).unsqueeze(3)
        iou_scores = self.compute_iou(grid_predicts, true_bboxs.unsqueeze(1).unsqueeze(1))
        iou_max = iou_scores.max(-1, keepdim=True)[0]
        noobj_mask = 1 - anchor_mask.squeeze(dim=2)
        label_noobj_mask = (iou_max < self.obj_thresh).squeeze(3).permute(0, 2, 1).float() * noobj_mask
        obj_conf_loss = anchor_mask.squeeze(dim=2) * self.focal_loss(p_conf, t_conf)
        noobj_conf_loss = label_noobj_mask * self.focal_loss(p_conf, t_conf)

        pos_count = anchor_mask.sum() + 1e-5
        neg_count = label_noobj_mask.sum() + 1e-5

        clas_loss = clas_loss.sum() * self.class_scale / batch_size
        coords_loss = coords_loss * self.coord_scale / batch_size
        obj_conf_loss = obj_conf_loss.sum() * self.obj_scale / batch_size
        noobj_conf_loss = noobj_conf_loss.sum() * self.noobj_scale / batch_size
        total_loss = clas_loss + coords_loss + obj_conf_loss + noobj_conf_loss

        # Online statistics
        print("network output shape: ({}, {}), pos_num: {}, neg_num: {}, total loss: {}, class loss: {}, coords loss: "
              "{}, obj_conf loss: {}, noobj_conf loss: {}".format(grid_w, grid_h, torch.floor(pos_count),
                                                                  torch.floor(neg_count), total_loss, clas_loss,
                                                                  coords_loss, obj_conf_loss, noobj_conf_loss))
        print("object average conf: {}".format((anchor_mask.squeeze(dim=2) * p_conf).sum() / pos_count))
        print("background average conf: {}".format((label_noobj_mask * p_conf).sum() / label_noobj_mask.sum()))

        return total_loss

    def rescale_to_img(self, coords, grid_h, grid_w):
        col_index = torch.arange(0, grid_w).repeat(grid_h, 1).view(grid_h * grid_w)
        row_index = torch.arange(0, grid_h).repeat(grid_w, 1).t().contiguous().view(grid_h * grid_h)
        img_coords = torch.zeros_like(coords)
        anchor_w = self.anchors[:, 0].contiguous().view(self.anchor_number, 1)
        anchor_h = self.anchors[:, 1].contiguous().view(self.anchor_number, 1)
        if self.use_gpu:
            anchor_w = anchor_w.cuda()
            anchor_h = anchor_h.cuda()
            col_index = col_index.cuda()
            row_index = row_index.cuda()
            img_coords = img_coords.cuda()
        # to img coords
        img_coords[:, :, 0, :] = (coords[:, :, 0, :] + col_index.float()) / grid_w * self.net_factor[0]
        img_coords[:, :, 1, :] = (coords[:, :, 1, :] + row_index.float()) / grid_h * self.net_factor[1]
        img_coords[:, :, 2, :] = coords[:, :, 2, :].exp() * anchor_w
        img_coords[:, :, 3, :] = coords[:, :, 3, :].exp() * anchor_h

        # to [x1, y1, x2, y2]
        img_coords[:, :, 0, :] = img_coords[:, :, 0, :] - img_coords[:, :, 2, :] / 2
        img_coords[:, :, 1, :] = img_coords[:, :, 1, :] - img_coords[:, :, 3, :] / 2
        img_coords[:, :, 2, :] = img_coords[:, :, 0, :] + img_coords[:, :, 2, :]
        img_coords[:, :, 3, :] = img_coords[:, :, 1, :] + img_coords[:, :, 3, :]
        return img_coords

    def encode_target(self, ground_truth, grid_h, grid_w):
        """
        每个图片的GT为[[x1, y1, x2, y2, class_index],[x1, y1, x2, y2, class_index],...]
        :param ground_truth:
        :param grid_h: 网络输出特征图的高度
        :param grid_w: 网络输出特征图的宽度
        :return:
        """
        batch_size = len(ground_truth)
        # anchor mask 标记哪些 anchor 激活负责检测目标, 负责检测目标置为1
        anchor_mask = torch.zeros(batch_size, self.anchor_number, grid_h * grid_w)
        # 构建yolo输出
        targets = torch.zeros(batch_size, self.anchor_number, 5 + 1, grid_h * grid_w)

        true_bboxs = torch.zeros((batch_size, self.max_box_per_image, 4))
        bbox_count = torch.zeros((batch_size,))

        if self.use_gpu:
            anchor_mask = anchor_mask.cuda()
            targets = targets.cuda()
            true_bboxs = true_bboxs.cuda()

        # 对于batch_size 内的每一个box, 分配anchor, 生成需要的格式
        for b in range(batch_size):
            bboxs = ground_truth[b]
            if len(bboxs) == 0:
                continue
            for box in bboxs:
                # 记录每张图片里的真实bbox
                bbox_ind = int(bbox_count[b] % self.max_box_per_image)
                true_bboxs[b][bbox_ind, :4] = box[:4]
                bbox_count[b] += 1

                best_iou = 0
                best_anchor_index = 0
                # Torch的多分类交叉熵函数的label输入为类别的index, 非 one-hot
                cls = box[4]
                # 匹配最佳anchor
                for ii, anchor in enumerate(self.anchors):
                    cur_iou = self.anchor_iou(box, anchor)
                    if cur_iou > best_iou:
                        best_iou = cur_iou
                        best_anchor_index = ii

                print("obj best iou: {}, best anchor index: {}".format(best_iou, best_anchor_index))
                anchor_w = self.anchors[best_anchor_index][0]
                anchor_h = self.anchors[best_anchor_index][1]
                # box中心坐标相对grid cell 左上角的偏移值以及bounding box 与anchor 的宽高比
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
                # 对于置信度, 可以让网络学习IOU, 也可以直接设置为1, 设置成IOU学习更难
                if self.hard_conf:
                    obj_conf = best_iou
                else:
                    obj_conf = torch.Tensor([1])
                # 将当前 box 的信息赋值给对应的anchor, 同时对 mask 进行标记
                grid_info = torch.cat([x_offset.view(-1, 1), y_offset.view(-1, 1), w_log.view(-1, 1), h_log.view(-1, 1), obj_conf.view(-1, 1), cls.view(-1, 1)], dim=1)
                targets[b, best_anchor_index, :, row * grid_w + col] = grid_info.clone()
                anchor_mask[b, best_anchor_index, row * grid_w + col] = 1

        return targets, anchor_mask, true_bboxs

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
        tl = torch.max(boxes1[..., :2], boxes2[..., :2])
        br = torch.min(boxes1[..., 2:], boxes2[..., 2:])
        wh = br - tl
        wh = torch.max(wh, torch.zeros_like(wh))
        inter = wh[..., 0] * wh[..., 1]
        area_1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
        area_2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

        return inter / (area_1 + area_2 - inter + 1e-5)

    def focal_loss(self, predict, target, alpha=1, gamma=2):
        bce_loss = F.binary_cross_entropy(predict, target, reduction='none')
        pt = torch.exp(-bce_loss)
        return alpha * (1 - pt) ** gamma * bce_loss
