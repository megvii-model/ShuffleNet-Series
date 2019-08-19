import os
import sys
import torch
import torch.nn as nn
import time
import logging
import argparse
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import cv2
import numpy as np
import PIL
from PIL import Image
from network import ShuffleNetV2
from utils import accuracy, AvgrageMeter, CrossEntropyLabelSmooth, save_checkpoint, get_lastest_model, get_parameters

class OpencvResize(object):

    def __init__(self, size=256):
        self.size = size

    def __call__(self, img):
        assert isinstance(img, PIL.Image.Image)
        img = np.asarray(img) # (H,W,3) RGB
        img = img[:,:,::-1] # 2 BGR
        img = np.ascontiguousarray(img)
        H, W, _ = img.shape
        target_size = (int(self.size/H * W + 0.5), self.size) if H < W else (self.size, int(self.size/W * H + 0.5))
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        img = img[:,:,::-1] # 2 RGB
        img = np.ascontiguousarray(img)
        img = Image.fromarray(img)
        return img

class ToBGRTensor(object):

    def __call__(self, img):
        assert isinstance(img, (np.ndarray, PIL.Image.Image))
        if isinstance(img, PIL.Image.Image):
            img = np.asarray(img)
        img = img[:,:,::-1] # 2 BGR
        img = np.transpose(img, [2, 0, 1]) # 2 (3, H, W)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        return img

class DataIterator(object):

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data[0], data[1]

class Lighting(object):

    def __init__(self, alphastd, eigval=None, eigvec=None):
        self.alphastd = alphastd
        if eigval is None:
            eigval = torch.Tensor([0.2175, 0.0188, 0.0045])
        if eigvec is None:
            eigvec = torch.Tensor([
                [-0.5675, 0.7192, 0.4009],
                [-0.5808, -0.0045, -0.8140],
                [-0.5836, -0.6948, 0.4203],
            ])
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        """
        :param img : (N,3,H,W) RGB
        """
        if self.alphastd == 0:
            return img

        device = img.device
        alpha = torch.normal(torch.zeros_like(self.eigval), self.alphastd)
        alpha = alpha.to(device)
        eigval = self.eigval.to(device)
        eigvec = self.eigvec.to(device)
        rgb = torch.mm(eigvec, eigval.mul(alpha).reshape(3,1)).squeeze() # (3)
        img = img.add(rgb.view(1, 3, 1, 1))
        return img

class ColorNormalize(object):

    def __init__(self, mean=None, std=None):
        if mean is None:
            mean = torch.Tensor([0.485, 0.456, 0.406])
        if std is None:
            std = torch.Tensor([0.229, 0.224, 0.225])
        self.mean = mean
        self.std = std

    def __call__(self, img):
        """
        :param img : (N,3,H,W) RGB
        """
        device = img.device
        mean = self.mean.to(device)
        std = self.std.to(device)
        img.sub_(mean.reshape(1, -1, 1, 1)).div_(std.reshape(1, -1, 1, 1))
        return img

def get_mean():
    from xml.dom.minidom import parse
    import numpy as np

    f = './ImageNet_1000_scale224_mean.xml'
    tree = parse(f)
    content = tree.documentElement
    data = content.getElementsByTagName('MeanImg')[0]
    data = data.getElementsByTagName('data')[0]
    mean = data.childNodes[0].data
    mean = mean.split(' ')
    res = []
    for m in mean:
        if m == '\n' or m == '':
            continue
        m = float(m[:-1]) if m.endswith('\n') else float(m)
        assert m <= 255
        res.append(m)
    mean = np.array(res).reshape((224,224,3)) # BGR
    mean = np.transpose(mean, [2, 0, 1])
    mean = mean[np.newaxis, ...]
    return mean

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', default=False, action='store_true')
    parser.add_argument('--eval-resume', type=str, default='./snetv2_residual_se.pkl', help='path for eval model')
    parser.add_argument('--batch-size', type=int, default=256, help='batch size')
    parser.add_argument('--total-iters', type=int, default=600000, help='total iters')
    parser.add_argument('--learning-rate', type=float, default=0.25, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight-decay', type=float, default=4e-5, help='weight decay')
    parser.add_argument('--save', type=str, default='./models', help='path for saving trained models')
    parser.add_argument('--label-smooth', type=float, default=0.1, help='label smoothing')

    parser.add_argument('--auto-continue', default=False, action='store_true', help='report frequency')
    parser.add_argument('--display-interval', type=int, default=20, help='report frequency')
    parser.add_argument('--val-interval', type=int, default=10000, help='report frequency')
    parser.add_argument('--save-interval', type=int, default=10000, help='report frequency')

    parser.add_argument('--train-dir', type=str, default='data/train', help='path to training dataset')
    parser.add_argument('--val-dir', type=str, default='data/val', help='path to validation dataset')

    args = parser.parse_args()
    return args


def main():
    args = get_args()

    # Log
    log_format = '[%(asctime)s] %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format, datefmt='%d %I:%M:%S')
    t = time.time()
    local_time = time.localtime(t)
    if not os.path.exists('./log'):
        os.mkdir('./log')
    fh = logging.FileHandler(os.path.join('log/train-{}{:02}{}'.format(local_time.tm_year % 2000, local_time.tm_mon, t)))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    use_gpu = False
    if torch.cuda.is_available():
        use_gpu = True

    assert os.path.exists(args.train_dir)
    train_dataset = datasets.ImageFolder(
        args.train_dir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ToTensor(),
        ])
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=1, pin_memory=use_gpu)
    train_dataprovider = DataIterator(train_loader)

    assert os.path.exists(args.val_dir)
    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(args.val_dir, transforms.Compose([
            OpencvResize(256),
            transforms.CenterCrop(224),
            ToBGRTensor(),
        ])),
        batch_size=200, shuffle=False,
        num_workers=1, pin_memory=use_gpu
    )
    val_dataprovider = DataIterator(val_loader)
    print('load data successfully')

    model = ShuffleNetV2()
    if args.eval:
        if args.eval_resume is not None:
            checkpoint = torch.load(args.eval_resume, map_location=None if use_gpu else 'cpu')
            print('==> Resuming from checkpoint..')
            load_checkpoint(model, checkpoint)

    optimizer = torch.optim.SGD(get_parameters(model), lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)
    criterion_smooth = CrossEntropyLabelSmooth(1000, 0.1)

    if use_gpu:
        model = nn.DataParallel(model)
        loss_function = criterion_smooth.cuda()
        device = torch.device("cuda")
    else:
        loss_function = criterion_smooth
        device = torch.device("cpu")

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                    lambda step : (1.0-step/args.total_iters) if step <= args.total_iters else 0, last_epoch=-1)

    model = model.to(device)

    all_iters = 0
    if args.auto_continue:
        lastest_model, iters = get_lastest_model()
        if lastest_model is not None:
            all_iters = iters
            checkpoint = torch.load(lastest_model, map_location=None if use_gpu else 'cpu')
            model.load_state_dict(checkpoint['state_dict'], strict=True)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
            print('load from checkpoint')

    args.optimizer = optimizer
    args.loss_function = loss_function
    args.scheduler = scheduler
    args.train_dataprovider = train_dataprovider
    args.val_dataprovider = val_dataprovider

    if args.eval:
        if args.eval_resume is not None:
            validate(model, device, args, all_iters=all_iters)
    else:
        while all_iters < args.total_iters:
            all_iters = train(model, device, args, val_interval=args.val_interval, all_iters=all_iters)
            validate(model, device, args, all_iters=all_iters)
        save_checkpoint({'state_dict': model.state_dict(),
                         'optimizer_state_dict': args.optimizer.state_dict(),
                         'lr_scheduler_state_dict': args.scheduler.state_dict()},
                        args.total_iters, tag='bnps-')


def train(model, device, args, *, val_interval, all_iters=None):

    optimizer = args.optimizer
    loss_function = args.loss_function
    scheduler = args.scheduler
    train_dataprovider = args.train_dataprovider

    t1 = time.time()
    Top1_err, Top5_err = 0.0, 0.0
    model.train()
    for iters in range(1, val_interval + 1):
        scheduler.step()
        all_iters += 1
        d_st = time.time()
        data, target = train_dataprovider.next()
        target = target.type(torch.LongTensor)
        data, target = data.to(device), target.to(device) # (N,3,H,W) RGB 0~1
        data = ColorNormalize()(Lighting(alphastd=0.1)(data))
        data = data.cpu().numpy()[:,::-1,:,:] # 2 BGR
        data = np.ascontiguousarray(data)
        data = torch.from_numpy(data).to(device)
        data_time = time.time() - d_st

        output_7, output_14, output_28, output_56 = model(data)
        loss = 1.0 * loss_function(output_7, target) + 0.7 * loss_function(output_14, target) + \
            0.5 * loss_function(output_28, target) + 0.3 * loss_function(output_56, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        prec1, prec5 = accuracy(output_7, target, topk=(1, 5))

        Top1_err += 1 - prec1.item() / 100
        Top5_err += 1 - prec5.item() / 100

        if all_iters % args.display_interval == 0:
            printInfo = 'TRAIN Iter {}: lr = {:.6f},\tloss = {:.6f},\t'.format(all_iters, scheduler.get_lr()[0], loss.item()) + \
                        'Top-1 err = {:.6f},\t'.format(Top1_err / args.display_interval) + \
                        'Top-5 err = {:.6f},\t'.format(Top5_err / args.display_interval) + \
                        'data_time = {:.6f},\ttrain_time = {:.6f}'.format(data_time, (time.time() - t1) / args.display_interval)
            logging.info(printInfo)
            t1 = time.time()
            Top1_err, Top5_err = 0.0, 0.0

        if all_iters % args.save_interval == 0:
            save_checkpoint({'state_dict': model.state_dict(),
                             'optimizer_state_dict': args.optimizer.state_dict(),
                             'lr_scheduler_state_dict': args.scheduler.state_dict()},
                            all_iters)

    return all_iters


def validate(model, device, args, *, all_iters=None):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    top5 = AvgrageMeter()

    loss_function = args.loss_function
    val_dataprovider = args.val_dataprovider

    mean = get_mean()
    mean = torch.from_numpy(mean).to(device).float() # (1, 3, 224, 224) BGR

    model.eval()
    max_val_iters = 250
    t1 = time.time()
    with torch.no_grad():
        for _ in range(1, max_val_iters + 1):
            data, target = val_dataprovider.next()
            target = target.type(torch.LongTensor)
            data, target = data.to(device), target.to(device) # data : BGR [0,255]
            data -= mean

            output = model(data)
            loss = loss_function(output, target)

            prec1, prec5 = accuracy(output, target, topk=(1, 5))
            n = data.size(0)
            objs.update(loss.item(), n)
            top1.update(prec1.item(), n)
            top5.update(prec5.item(), n)

    logInfo = 'TEST Iter {}: loss = {:.6f},\t'.format(all_iters, objs.avg) + \
              'Top-1 err = {:.6f},\t'.format(1 - top1.avg / 100) + \
              'Top-5 err = {:.6f},\t'.format(1 - top5.avg / 100) + \
              'val_time = {:.6f}'.format(time.time() - t1)
    logging.info(logInfo)



def load_checkpoint(net, checkpoint):
    if 'state_dict' in checkpoint:
        checkpoint = dict(checkpoint['state_dict'])
    for k in checkpoint:
        if 'module' in k:
            checkpoint[k[7:]] = checkpoint.pop(k)
    for name, param in net.named_parameters():
        if name not in checkpoint:
            if 'predict' not in name:
                print(name)
        else:
            param.data = checkpoint[name].data
    for name, buffer in net.named_buffers():
        if name not in checkpoint:
            if 'predict' not in name:
                print(name)
        else:
            buffer.data = checkpoint[name].data


if __name__ == "__main__":
    main()

