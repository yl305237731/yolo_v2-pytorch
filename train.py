import math
import time
import datetime
import argparse
import os
import torch
from vision.network import YoloV2
from vision.loss import YoloLossLayer
from tools.dataloader import VOCDataSet
from tools.augment import VOCDataAugmentation
from torch.utils.data import DataLoader
from torchvision import transforms
from tools.util import custom_collate_fn, adjust_learning_rate


parser = argparse.ArgumentParser("--------Train YOLO-V2--------")
parser.add_argument('--weights_save_folder', default='./weights', type=str, help='Dir to save weights')
parser.add_argument('--imgs_dir', default='./data/imgs', help='train images dir')
parser.add_argument('--annos_dir', default='./data/xmls', type=str, help='annotation xml dir')
parser.add_argument('--batch_size', default=16, type=int, help="batch size")
parser.add_argument('--net_w', default=416, type=int, help="input image width")
parser.add_argument('--net_h', default=416, type=int, help="input image height")
parser.add_argument('--anchors', default=[], type=list, help="anchor size[w, h]")
parser.add_argument('--max_epoch', default=30, type=int, help="max training epoch")
parser.add_argument('--initial_lr', default=1e-3, type=float, help="initial learning rate")
parser.add_argument('--gamma', default=0.1, type=float, help="gamma for adjust lr")
parser.add_argument('--weight_decay', default=5e-4, type=float, help="weights decay")
parser.add_argument('--decay1', default=190, type=int)
parser.add_argument('--decay2', default=200, type=int)
parser.add_argument('--num_workers', default=2, type=int)
parser.add_argument('--num_gpu', default=1, type=int, help="gpu number")
parser.add_argument('--pre_train', default=False, type=bool, help="whether use pre-train weights for change class number")
args = parser.parse_args()


labels = [""]


def train(net, optimizer, trainSet, use_gpu):
    net.train()
    epoch = 0
    print('Loading Dataset...')

    epoch_size = math.ceil(len(trainSet) / args.batch_size)
    max_iter = args.max_epoch * epoch_size

    stepvalues = (args.decay1 * epoch_size, args.decay2 * epoch_size)
    step_index = 0
    start_iter = 0

    print("Begin training...")
    for iteration in range(start_iter, max_iter):
        if iteration % epoch_size == 0:
            epoch += 1
            batch_iterator = iter(DataLoader(trainSet, args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=custom_collate_fn))
            if epoch % 10 == 0 and epoch > 0:
                if args.num_gpu > 1:
                    torch.save(net.module.state_dict(), os.path.join(args.weights_save_folder, 'epoch_' + str(epoch) + '.pth'))
                else:
                    torch.save(net.state_dict(), os.path.join(args.weights_save_folder, 'epoch_' + str(epoch) + '.pth'))
        load_t0 = time.time()
        if iteration in stepvalues:
            step_index += 1
        lr = adjust_learning_rate(args.initial_lr, optimizer, args.gamma, epoch, step_index, iteration, epoch_size)

        images, targets = next(batch_iterator)

        if use_gpu:
            images = images.cuda()

        out = net(images)
        optimizer.zero_grad()
        loss = criterion(out, targets)
        loss.backward()
        optimizer.step()

        load_t1 = time.time()
        batch_time = load_t1 - load_t0
        eta = int(batch_time * (max_iter - iteration))
        print('Epoch:{}/{} || Epochiter: {}/{} || Iter: {}/{} || Loss: {:.4f}|| LR: {:.8f} || Batchtime: {:.4f} s ||'
              ' ETA: {}'.format(epoch, args.max_epoch, (iteration % epoch_size) + 1, epoch_size, iteration + 1,
                                max_iter, loss, lr, batch_time, str(datetime.timedelta(seconds=eta))))
    if args.num_gpu > 1:
        torch.save(net.module.state_dict(), os.path.join(args.weights_save_folder, 'voc.pth'))
    else:
        torch.save(net.state_dict(), os.path.join(args.weights_save_folder, 'voc.pth'))
    print('Finished Training')


if __name__ == '__main__':

    use_gpu = torch.cuda.is_available()
    net = YoloV2(class_num=len(labels), anchor_num=len(args.anchors))

    if args.pre_train:
        device = torch.device("cuda" if use_gpu else "cpu")
        pretrained_dict = torch.load(os.path.join(args.weights_save_folder, "Final.pth"), map_location=torch.device(device))
        model_dict = net.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        net.load_state_dict(model_dict)

    if args.num_gpu > 1 and use_gpu:
        net = torch.nn.DataParallel(net).cuda()
    elif use_gpu:
        net = net.cuda()

    if not os.path.exists(args.weights_save_folder):
        os.mkdir(args.weights_save_folder)

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    augmentation = VOCDataAugmentation()
    trainSet = VOCDataSet(img_dir=args.imgs_dir, xml_dir=args.annos_dir, name_list=labels, shuffle=True,
                          transform=transform, augmentation=augmentation)
   
    criterion = YoloLossLayer(anchors=args.anchors, class_number=len(labels), reduction=32, use_gpu=use_gpu,
                              hard_conf=False)
    optimizer = torch.optim.Adam(net.parameters(), lr=args.initial_lr, weight_decay=args.weight_decay)
    train(net, optimizer, trainSet, use_gpu)
