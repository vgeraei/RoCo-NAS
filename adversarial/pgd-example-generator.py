'''Inference CIFAR10 with PyTorch.'''
from __future__ import print_function

import sys
# update your projecty root path before running
sys.path.insert(0, 'C:\\Users\\Sima\\codes\\NSGA\\nsga-net')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision

import os
import sys
import time
import logging
import argparse
import pickle
import numpy as np
import torchattacks
from torchattacks import PGD, FGSM, CW, BIM, DeepFool

from misc import utils
import torch.optim as optim

# model imports
from models import macro_genotypes
from models.macro_models import EvoNetwork
import models.micro_genotypes as genotypes
from models.micro_models import PyramidNetworkCIFAR as PyrmNASNet

from art.attacks.evasion import FastGradientMethod, ProjectedGradientDescent
from art.estimators.classification import PyTorchClassifier
from art.utils import load_cifar10

device = 'cuda'
args = 0

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def main():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Testing')
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument('--data', type=str, default='../data', help='location of the data corpus')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
    parser.add_argument('--save', type=str, default='PGD-test', help='experiment name')
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

    args.save = '{}-{}'.format(args.save + '-' + args.arch, time.strftime("%Y%m%d-%H%M%S"))
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

    cudnn.enabled = False
    cudnn.benchmark = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)

    # Data
    _, valid_transform = utils._data_transforms_cifar10(args)

    train_transform, valid_transform = utils._data_transforms_cifar10(args)
    train_data = torchvision.datasets.CIFAR10(root=args.data, train=True, download=True, transform=train_transform)

    train_queue = torch.utils.data.DataLoader(
        train_data, batch_size=args.batch_size, shuffle=True, pin_memory=True, num_workers=2)

    valid_data = torchvision.datasets.CIFAR10(root=args.data, train=False, download=True, transform=valid_transform)
    # valid_data = torch.tensor(valid_data, dtype=torch.float)

    valid_queue = torch.utils.data.DataLoader(
        valid_data, batch_size=args.batch_size, shuffle=False, pin_memory=True, num_workers=2)


    # valid_queue = torch.tensor(valid_queue, dtype=torch.float)
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

    # net = net.double()

    criterion = nn.CrossEntropyLoss()
    criterion.to(device)

    # print(type(net.model))

    # Step 1: Load the cifar10 dataset

    (x_train, y_train), (x_test, y_test), min_pixel_value, max_pixel_value = load_cifar10()
    x_train, y_train = x_train[:10000], y_train[:10000]
    x_test, y_test = x_test[:1000], y_test[:1000]
    im_shape = x_train[0].shape

    # print(type(x_test))
    test_data = []
    for i in range(0, len(x_test)):
        test_data.append(valid_transform(np.array(x_test[i])))
    # a = np.array(torch.tensor(x_test[0]))
    # a = valid_transform(image)

    # test_data_numpy = np.array(test_data)
    test_data = [t.numpy() for t in test_data]
    test_data_float = [t.astype('float32') for t in test_data]

    # Step 1a: Swap axes to PyTorch's NCHW format

    x_train = np.swapaxes(x_train, 1, 3).astype(np.float32)
    x_test = np.swapaxes(x_test, 1, 3).astype(np.float32)
    # test_data_numpy = np.swapaxes(test_data_numpy, 1, 3).astype(np.float32)

    # Step 2: Create the model

    model = Net()

    # Step 2a: Define the loss function and the optimizer

    criterion2 = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # Step 3: Create the ART classifier


    classifier_std = PyTorchClassifier(
        model=net,
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion2,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
    )

    classifier_float = PyTorchClassifier(
        model=net.float(),
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion2,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
    )

    classifier_double = PyTorchClassifier(
        model=net.double(),
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion2,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
    )

    classifier_dummy = PyTorchClassifier(
        model=model,
        clip_values=(min_pixel_value, max_pixel_value),
        loss=criterion2,
        optimizer=optimizer,
        input_shape=(3, 32, 32),
        nb_classes=10,
    )

    # print(type(model))
    # classifier.fit(x_train, y_train, batch_size=64, nb_epochs=30)

    # inference on original CIFAR-10 test images
    # infer(valid_queue, net, criterion)

    # Step 5: Evaluate the ART classifier on benign test examples

    predictions = classifier_double.predict(test_data)
    accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))

    # Step 6: Generate adversarial test examples

    # attack = ProjectedGradientDescent(estimator=classifier_double, eps=0.2)
    # x_test_adv = attack.generate(x=test_data_float)
    # pgd_attack = PGD(net.float(), eps=1 / 255, alpha=2 / 255, steps=1)
    # pgd_attack = CW(net.float(), steps=100)
    # pgd_attack = FGSM(net.float(), eps=0.3)
    pgd_attack = PGD(net.float(), eps=8 / 255, steps=40)
    output = open('pgd-examples-' + args.arch + '.pkl', 'wb')
    pickle.dump(pgd_attack, output)
    output.close()

    # Step 7: Evaluate the ART classifier on adversarial test examples

    net.eval()

    correct = 0
    total = 0



    for images, labels in valid_queue:
        images = images.float()
        # labels = labels
        # print(type(labels))
        images = pgd_attack(images, labels).cuda()
        outputs = net(images)

        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels.cuda()).sum()

    print('Accuracy of Adversarial images: %f %%' % (100 * float(correct) / total))
    logging.info("PGD Accuracy: %f", (100 * float(correct) / total))
    logging.info("Total: %f", total)

    # predictions = classifier_float.predict(x_test_adv)
    # accuracy = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    # print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))


# def infer(valid_queue, net, criterion):
#     net.eval()
#     test_loss = 0
#     correct = 0
#     total = 0
#
#     with torch.no_grad():
#         for step, (inputs, targets) in enumerate(valid_queue):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs, _ = net(inputs)
#             loss = criterion(outputs, targets)
#
#             test_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
#
#             if step % args.report_freq == 0:
#                 logging.info('valid %03d %e %f', step, test_loss/total, 100.*correct/total)
#
#     acc = 100.*correct/total
#     logging.info('valid acc %f', acc)
#
#     return test_loss/total, acc


if __name__ == '__main__':
    main()