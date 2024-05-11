import logging
import time
from datetime import datetime
import argparse
from torch.backends import cudnn
import torch
import dataset
from model.resnet18_model import resnet18_model
from algorithm.utils import AverageMeter
from algorithm.utils import accuracy

# Save log file
log_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
log_path = "made_labelset_model/" + log_filename + ".log"
file_handler = logging.FileHandler(log_path)
logging.basicConfig(format='[%(asctime)s] - %(message)s', datefmt='%Y/%m/%d %H:%M:%S', level=logging.DEBUG, handlers=[logging.StreamHandler(), file_handler])
print("Log File save in : " + log_path)

# parameters
parser = argparse.ArgumentParser(description='')
parser.add_argument('--epochs', default=200, type=int, 
                    help='number of total epochs')
parser.add_argument('--batch_size', default=64, type=int, 
                    help='batch size')
parser.add_argument('--dataset', default='cifar10', type=str, choices=['cifar10', 'cifar100_SM', 'cifar100_T'], 
                    help='dataset name')
parser.add_argument('--model', default='resnet18', type=str, choices=['resnet18'], 
                    help='training model name')
parser.add_argument('--lr', default=0.01, type=float, 
                    help='learning rate')
parser.add_argument('--momentum', default=0.9, type=float, 
                    help='momentum')
parser.add_argument('--wd', default=1e-3, type=float, 
                    help='weight decay')

parser.add_argument('--target_acc', default=97.86, type=float, 
                    help='= delta2*100, accuracy of early stopping')
parser.add_argument('--deviation', default=0.1, type=float, 
                    help='acceptable deviation value for early stopping')
args = parser.parse_args()
logging.info(args)

def main():
    cudnn.benchmark = True
    # dataset
    if args.dataset == "cifar10":
        train_loader, test_loader = dataset.cifar10_dataloader(batch_size=args.batch_size, creation_method='APLG')
        num_class=10
    elif args.dataset == 'cifar100_SM':
        train_loader, test_loader = dataset.cifar100_small_mammals_dataloader(batch_size=args.batch_size, creation_method='APLG')
        num_class=5
    elif args.dataset == 'cifar100_T':
        train_loader, test_loader = dataset.cifar100_trees_dataloader(batch_size=args.batch_size, creation_method='APLG')
        num_class=5
    else:
        assert "Unknown dataset"
    
    # model
    if args.model == 'resnet18':
        model = resnet18_model(num_class)
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
        trainacc, trainloss = train(epoch, train_loader, model, optimizer)
        train_acc_history.append(trainacc)
        train_loss_history.append(trainloss)
        scheduler.step()
        testacc, testloss = test(test_loader, model)
        test_acc_history.append(testacc)
        test_loss_history.append(testloss)
        if trainacc > args.target_acc - args.deviation:
            logging.info('Stopping training. Train accuracy is within +/- {0} of the target probability ({1})!!'.format(args.deviation, args.target_acc))
            break
    
    logging.info('Max testing accuracy: {0} in {1} epoch!!!'.format(max(test_acc_history), test_acc_history.index(max(test_acc_history))))
    model_path = "made_labelset_model/" + args.dataset + "_" + args.model + "_" + log_filename + ".pt"
    torch.save(model.state_dict(), model_path)
    logging.info('Model save in {0}!!!'.format(model_path))

# Training
def train(epoch, train_loader, model, optimizer):
    data_time = AverageMeter()
    batch_time = AverageMeter()
    losses = AverageMeter()
    acc = AverageMeter()
    batch_end = time.time()
    end = time.time()
    loss_CE = torch.nn.CrossEntropyLoss().cuda()

    model.train()
    for i, (x_ori, x_aug1, x_aug2, y, y_true, index) in enumerate(train_loader):
        data_time.update(time.time() - batch_end)

        # The model will be trained using standard supervised learning methods
        # x_aug1, x_aug2, y will not be used here
        # eval_only=True is to save time, because the other three outputs will not be used here
        x_ori, y_true = x_ori.cuda(), y_true.to(torch.long).cuda()
        output = model(x_ori, eval_only=True)

        # Total loss
        final_loss = loss_CE(output, y_true)

        # training accuracy
        y_true =  y_true.cuda()
        final_acc = accuracy(output, y_true)[0]

        optimizer.zero_grad()
        final_loss.backward()
        optimizer.step()

        # measure accuracy and record loss
        losses.update(final_loss.item(), x_ori.size(0))
        acc.update(final_acc.item())
        # measure elapsed time
        batch_time.update(time.time() - batch_end)
        batch_end = time.time()

        if i % 50 == 0 or i == (len(train_loader)-1):
            logging.info('Epoch: [{0}][{1}/{2}]\t'
                        'Time {3} ({batch_time.avg:.3f})\t'
                        'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                        'Accuracy {acc.val:.4f} ({acc.avg:.4f})\t'.format(
                epoch, i, len(train_loader), time.time()-end, batch_time=batch_time,
                data_time=data_time, loss=losses, acc=acc))
            end = time.time()
    return acc.avg, losses.avg

# Testing
def test(test_loader, model):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    end = time.time()

    loss_CE = torch.nn.CrossEntropyLoss().cuda()
    model.eval()
    with torch.no_grad():
        for i, (input, target) in enumerate(test_loader):
            input, target = input.cuda(), target.cuda()

            # Prediction Test Data
            output = model(input, eval_only=True)
            loss = loss_CE(output, target)

            output = output.float()
            loss = loss.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % 50 == 0 or i == (len(test_loader)-1):
                logging.info('Test: [{0}/{1}]\t'
                            'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                    i, len(test_loader), batch_time=batch_time, loss=losses,top1=top1))

    logging.info(' * Prec@1 {top1.avg:.3f}'.format(top1=top1))
    return top1.avg, losses.avg

if __name__ == '__main__':
    time_start = time.time()
    main()
    time_end = time.time()
    MM,SS = divmod(round(time_end-time_start, 2), 60)
    HH,MM = divmod(MM, 60)
    logging.info('run time %02d:%02d:%02d'% (HH, MM, SS))