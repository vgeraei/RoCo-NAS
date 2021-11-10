'''Inference CIFAR10 with PyTorch.'''
from __future__ import print_function

import sys

# update your project root path before running
sys.path.insert(0, 'C:\\Users\\Sima\\codes\\NSGA\\nsga-net')

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision

import os
import time
import logging
import argparse
import numpy as np
import pickle

from misc import utils

# model imports
from models import macro_genotypes
from models.macro_models import EvoNetwork
import models.micro_genotypes as genotypes
from models.micro_models import PyramidNetworkCIFAR as PyrmNASNet

import matplotlib.pyplot as plt
import matplotlib.image as mpimg


device = 'cuda'
args = 0


def main():
    global args
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Testing')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--save', type=str, default='EXP', help='experiment name')
    parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
    parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
    parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
    parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
    parser.add_argument('--layers', default=20, type=int, help='total number of layers (default: 20)')
    parser.add_argument('--droprate', default=0, type=float, help='dropout probability (default: 0.0)')
    parser.add_argument('--init_channels', type=int, default=32, help='num of init channels')
    parser.add_argument('--arch', type=str, default='NSGANet', help='which architecture to use')
    parser.add_argument('--filter_increment', default=4, type=int, help='# of filter increment')
    parser.add_argument('--SE', action='store_true', default=False, help='use Squeeze-and-Excitation')
    parser.add_argument('--model_path', type=str, default='EXP/model.pt', help='path of pretrained model')
    parser.add_argument('--net_type', type=str, default='micro', help='(options)micro, macro')

    args = parser.parse_args()

    args.save = 'Acc-{}-{}'.format(args.save, args.arch + '-' + time.strftime("%Y%m%d-%H%M%S"))
    utils.create_exp_dir(args.save)

    global device

    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    if not torch.cuda.is_available():
        logging.info('no gpu device available')
        sys.exit(1)

    if args.auxiliary and args.net_type == 'macro':
        logging.info('auxiliary head classifier not supported for macro search space models')
        sys.exit(1)

    logging.info("args = %s", args)

    cudnn.enabled = True
    cudnn.benchmark = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    # Data
    _, valid_transform = utils._data_transforms_cifar10(args)


    my_x = []
    my_y = []

    my_dataset = pickle.load(open("adversarial_dataset.pkl", "rb"))

    valid_data = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)

    valid_queue_og = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    # valid_queue = torch.utils.data.DataLoader(
    #     my_dataset, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)

    # print("Original data: ", valid_data)
    # print("My data: ", my_dataset)

    # valid_queue = my_dataloader

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model

    genome = eval("macro_genotypes.%s" % args.arch)
    channels = [(3, 128), (128, 128), (128, 128)]
    net = EvoNetwork(genome, channels, 10, (32, 32), decoder='dense')

    # logging.info("{}".format(net))
    logging.info("param size = %fMB", utils.count_parameters_in_MB(net))

    net = net.to(device)
    # no drop path during inference
    net.droprate = 0.0
    utils.load(net, args.model_path)

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    # inference on original CIFAR-10 test images
    infer(valid_queue, net, criterion)


def infer(valid_queue, net, criterion):
    global args
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(valid_queue):
            inputs, targets = inputs.to(device), targets.to(device)

            # print("Inputs:")
            # print(inputs.size())
            # print("Targets:")
            # print(targets.long())

            targets = targets.long()

            # outputs, _ = net(inputs.permute(0, 3, 1, 2))
            outputs = net(inputs)
            # print(outputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            if step % args.report_freq == 0:
                logging.info('valid %03d %e %f', step, test_loss / total, 100. * correct / total)

    acc = 100. * correct / total
    logging.info('valid acc %f', acc)

    return test_loss / total, acc


if __name__ == '__main__':
    # adversarial_examples = pickle.load(open("dqa/Pixel_CMAES/Cifar10/LeNet/results_1.pkl", "rb"))
    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    # columns = ['attack', 'model',
    #                         'threshold', 'image',
    #                         'true', 'predicted',
    #                         'success', 'cdiff',
    #                         'prior_probs', 'predicted_probs',
    #                         'perturbation', 'attacked_image',
    #                         'l2_distance']
    # img_data = adversarial_examples[50]
    # img = img_data[11]
    # img = np.array(img, dtype='uint8')
    # true_class = img_data[4]
    # predicted_class = img_data[5]
    # print(img_data)
    # print(classes[true_class], classes[predicted_class])
    # imgplot = plt.imshow(img)
    # plt.show()

    main()

