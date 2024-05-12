import copy
import time
import logging
import time
import numpy as np
import copy
import torch
from torch import nn
import torch.nn.functional as F
from utils import InfoNCE, AverageMeter, accuracy

# FATEL framework
class FATEL():
    def __init__(self, train_loader, pretrain_epoch=10):
        self.loss_CrossEntropy = nn.CrossEntropyLoss().cuda()
        self.loss_InfoNCE = InfoNCE().cuda()
        # Prediction results in the previous epoch
        self.pre_prediction = torch.zeros_like(train_loader.dataset.labelset).float().cuda()
        # warmup and phi
        self.warmup = pretrain_epoch

    # training
    def train(self, epoch, train_loader, model, optimizer):
        data_time = AverageMeter()
        batch_time = AverageMeter()
        losses = AverageMeter()
        CE_Losses = AverageMeter()
        acc = AverageMeter()
        batch_end = time.time()
        end = time.time()

        model.train()
        for i, (x_ori, x_aug1, x_aug2, y, y_true, index) in enumerate(train_loader):
            data_time.update(time.time() - batch_end)
            batch_size = y.size()[0]

            x_ori, x_aug1, x_aug2, y = x_ori.cuda(), x_aug1.cuda(), x_aug2.cuda(), y.float().cuda()
            feature_pred_ori, feature_pred_aug1, feature_pred_aug2, y_pred_ori = model(x_ori=x_ori, x_aug1=x_aug1, x_aug2=x_aug2)
            y_pred_ori_probas = torch.softmax(y_pred_ori, dim=-1)

            # InfoNCE Loss
            consist_loss_aug1 = self.loss_InfoNCE(feature_pred_ori, feature_pred_aug1)
            consist_loss_aug2 = self.loss_InfoNCE(feature_pred_ori, feature_pred_aug2)

            # Cross Entropy Loss
            y_new = torch.logical_or(y.bool(), self.pre_prediction[index].bool()).float()
            super_loss = -torch.mean(torch.sum(torch.log(1.0000001 - y_pred_ori_probas) * (1 - y_new), dim=1))
            # After Warm-up ends, start adding the prediction of the previous epoch to the candidate labelset of this epoch
            if epoch >= self.warmup:
                pi = torch.argmax(y_pred_ori_probas, dim=1)
                self.pre_prediction[index, :] = F.one_hot(pi, y.size()[1]).float()
            
            # Mixup
            Eta = np.random.beta(8.0, 8.0)
            idx_rp = torch.randperm(batch_size)
            x_aug1_mix = Eta * x_aug1 + (1 - Eta) * x_aug1[idx_rp]
            y_mix = torch.logical_or(y.bool(), y[idx_rp].bool()).float()
            
            y_pred_aug1_mix =  model(x_ori=x_aug1_mix, eval_only=True)
            y_pred_aug1_mix_probas = torch.softmax(y_pred_aug1_mix, dim=-1)
            mixup_loss = -torch.mean(torch.sum(torch.log(1.0000001 - y_pred_aug1_mix_probas) * (1 - y_mix), dim=1))

            # Final loss
            final_loss = consist_loss_aug1 + consist_loss_aug2 + super_loss + mixup_loss
            if torch.any(torch.isnan(final_loss)):
                raise ValueError("Log_prob has nan!")

            # training accuracy and CEloss
            # Only used to inform users and not used for training
            y_true =  y_true.cuda()
            final_acc = accuracy(y_pred_ori_probas, y_true)[0]
            CE_Loss = self.loss_CrossEntropy(y_pred_ori, y_true.to(torch.int64))

            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            losses.update(final_loss.item(), x_ori.size(0))
            acc.update(final_acc.item())
            CE_Losses.update(CE_Loss.item(), x_ori.size(0))
            batch_time.update(time.time() - batch_end)
            batch_end = time.time()

            if i % 50 == 0 or i == (len(train_loader)-1):
                logging.info('Epoch: [{0}][{1}/{2}]\t'
                            'Time {3} ({batch_time.avg:.3f})\t'
                            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'CE_Loss {CE_Loss.val:.4f} ({CE_Loss.avg:.4f})\t'
                            'Accuracy {acc.val:.4f} ({acc.avg:.4f})\t'.format(
                    epoch, i, len(train_loader), time.time()-end, batch_time=batch_time,
                    data_time=data_time, loss=losses, CE_Loss=CE_Losses, acc=acc))
                end = time.time()
        
        return acc.avg, losses.avg

    # testing
    def test(self, test_loader, model):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()

        model.eval()
        with torch.no_grad():
            for i, (input, target) in enumerate(test_loader):
                input, target = input.cuda(), target.cuda()
                output = model(x_ori=input, eval_only=True)

                loss = self.loss_CrossEntropy(output, target)
                output = output.float()
                loss = loss.float()

                prec1 = accuracy(output.data, target)[0]
                losses.update(loss.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))

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