import os
import logging
import time
from datetime import datetime
import argparse
from torch.backends import cudnn
import torch
import dataset
from model.Resnet18 import Resnet18
from model.wideresnet import WideResNet
from FATEL import FATEL
from FAPT import FAPT

# Save log file
if not os.path.exists('log'):
    os.makedirs('log')
log_filename = datetime.now().strftime('%Y%m%d_%H%M%S')
log_path = 'log/' + log_filename + '.log'
file_handler = logging.FileHandler(log_path)
logging.basicConfig(format='[%(asctime)s] - %(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.DEBUG, handlers=[logging.StreamHandler(), file_handler])
print('Log File save in : ' + log_path)

# parameters
parser = argparse.ArgumentParser(description='')
parser.add_argument('--epochs', default=200, type=int, 
                    help='number of total epochs')
parser.add_argument('--batch_size', default=64, type=int, 
                    help='batch size')
parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100_SM', 'cifar100_T', 'cifar100_SM_500', 'cifar100_T_500'], 
                    help='dataset name')
parser.add_argument('--model', default='resnet18', type=str, choices=['wideresnet', 'resnet18'], 
                    help='training model name')
parser.add_argument('--lr', default=0.01, type=float, 
                    help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, 
                    help='momentum')
parser.add_argument('--wd', default=1e-3, type=float, 
                    help='weight decay')

parser.add_argument('--framework', default='FAPT', type=str, choices=['FAPT', 'FATEL'], 
                    help='UPLL framework name')
parser.add_argument('--creation_method', default='CLIG', type=str, choices=['CLIG', 'APLG'], 
                    help='How to generate a data set')
parser.add_argument('--partial_rate', default=0.1, type=float, 
                    help='For APLG. The probability that each class (except the true label) will be added to the candidate labelset.')
parser.add_argument('--noisy_rate', default=0.3, type=float, 
                    help='For APLG. The probability that the true label is not in the candidate labelset.')

parser.add_argument('--warm_up', default=10, type=int, 
                    help='= R, number of warm-up epochs')
parser.add_argument('--phi', default=0.9, type=float, 
                    help='= phi, pseudo target update ratio')
parser.add_argument('--data_ratio', default=1.0, type=float, 
                    help='For the experiment of reduce the amount of data. Control the amount of training data.')
args = parser.parse_args()
logging.info(args)

def main():
    args = parser.parse_args()
    cudnn.benchmark = True
    # dataset
    if args.dataset == "cifar10":
        train_loader, test_loader = dataset.cifar10_dataloader(batch_size=args.batch_size, creation_method=args.creation_method, partial_rate=args.partial_rate, noisy_rate=args.noisy_rate, data_ratio=args.data_ratio)
        num_class=10
    elif args.dataset == 'cifar100_SM':
        train_loader, test_loader = dataset.cifar100_small_mammals_dataloader(batch_size=args.batch_size, creation_method=args.creation_method, partial_rate=args.partial_rate, noisy_rate=args.noisy_rate)
        num_class=5
    elif args.dataset == 'cifar100_T':
        train_loader, test_loader = dataset.cifar100_trees_dataloader(batch_size=args.batch_size, creation_method=args.creation_method, partial_rate=args.partial_rate, noisy_rate=args.noisy_rate)
        num_class=5
    elif args.dataset == 'cifar100_SM_500':
        train_loader, test_loader = dataset.cifar100_small_mammals_part_dataloader(batch_size=args.batch_size, creation_method=args.creation_method, partial_rate=args.partial_rate, noisy_rate=args.noisy_rate)
        num_class=5
    elif args.dataset == 'cifar100_T_500':
        train_loader, test_loader = dataset.cifar100_trees_part_dataloader(batch_size=args.batch_size, creation_method=args.creation_method, partial_rate=args.partial_rate, noisy_rate=args.noisy_rate)
        num_class=5
    else:
        assert "Unknown dataset"
    
    # framework
    if args.framework == 'FAPT':
        framework = FAPT(train_loader, args.warm_up, args.phi)
    elif args.framework == 'FATEL':
        framework = FATEL(train_loader, args.warm_up)
    else:
        assert "Unknown framework"

    # model
    if args.model == 'resnet18':
        model = Resnet18(num_class)
    elif args.model == 'wideresnet':
        model = WideResNet(34, num_class, widen_factor=10, dropRate=0.0)
    else:
        assert "Unknown model"
    model = model.cuda()

    # optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    # scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], last_epoch=-1)

    train_acc_history = []
    train_loss_history = []
    test_acc_history = []
    test_loss_history = []
    # Train loop
    print("Start training!!")
    for epoch in range(0, args.epochs):
        trainacc, trainloss = framework.train(epoch, train_loader, model, optimizer)
        train_acc_history.append(trainacc)
        train_loss_history.append(trainloss)
        scheduler.step()
        testacc, testloss = framework.test(test_loader, model)
        test_acc_history.append(testacc)
        test_loss_history.append(testloss)
    
    logging.info('Max testing accuracy: {0} in {1} epoch!!!'.format(max(test_acc_history), test_acc_history.index(max(test_acc_history))))


if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    MM,SS = divmod(round(time_end-time_start, 2), 60)
    HH,MM = divmod(MM, 60)
    logging.info('run time %02d:%02d:%02d'% (HH, MM, SS))