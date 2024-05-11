import copy
import time
import logging
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from algorithm.utils import AverageMeter
from algorithm.utils import accuracy
from algorithm.utils_loss import InfoNCE
# from algorithm.utils_loss import InfoNCE2

class Test14():
    def __init__(self, train_loader, minimode = 1, pretrain_epoch=10, phi=0.9):
        self.loss_CrossEntropy = nn.CrossEntropyLoss().cuda()
        self.loss_KLDiv = nn.KLDivLoss(reduction='batchmean').cuda()
        self.loss_InfoNCE = InfoNCE().cuda()

        # Test14-1, Test14-2
        confidence = copy.deepcopy(train_loader.dataset.labelset)
        confidence = confidence / confidence.sum(axis=1)[:, None]
        self.pseudo_target = confidence.cuda()
        # Test14-3, Test14-4
        self.y_tmp = torch.zeros_like(train_loader.dataset.labelset).float().cuda()
        # 前半不更新pseudo_target或y_tmp
        self.warmup = pretrain_epoch

        self.minimode = minimode # Test14-1 or Test14-2 or Test14-3 or Test14-4 or Test14-5
        self.phi = phi
        logging.info("Test14-"+str(self.minimode)+"!!!")

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
            # tt = time.time()
            # 執行時間
            data_time.update(time.time() - batch_end)
            batch_size = y.size()[0]
            # print(i)
            x_ori, x_aug1, x_aug2, y = x_ori.cuda(), x_aug1.cuda(), x_aug2.cuda(), y.float().cuda()
            # print(x_ori)
            
            # feature_pred_ori = model.forward_1(x_ori)
            # feature_pred_aug1 = model.forward_1(x_aug1)
            # feature_pred_aug2 = model.forward_1(x_aug2)

            # y_pred_ori = model.forward_2(x_ori)
            feature_pred_ori, feature_pred_aug1, feature_pred_aug2, y_pred_ori = model(x_ori=x_ori, x_aug1=x_aug1, x_aug2=x_aug2)
            y_pred_ori_probas = torch.softmax(y_pred_ori, dim=-1)
            # print(y_true)
            # print(torch.argmax(y_pred_ori,dim=1))
            # max_pred_indices = torch.argmax(y_pred_ori, dim=1, keepdim=True)

            if self.minimode == 1 or self.minimode == 2:
                super_loss = -torch.mean(torch.sum(torch.log(y_pred_ori_probas) * self.pseudo_target[index], dim=1))

                # mixup
                Eta = np.random.beta(8.0, 8.0)
                idx_rp = torch.randperm(batch_size)
                x_aug1_mix = Eta * x_aug1 + (1 - Eta) * x_aug1[idx_rp]
                pseudo_target_tmp = self.pseudo_target[index]
                pseudo_target_mix = Eta * pseudo_target_tmp + (1 - Eta) * pseudo_target_tmp[idx_rp]
                # y_pred_aug1_mix = model.forward_2(x_aug1_mix)
                y_pred_aug1_mix =  model(x_ori=x_aug1_mix, eval_only=True)
                y_pred_aug1_mix_probas = torch.softmax(y_pred_aug1_mix, dim=-1)
                mixup_loss = -torch.mean(torch.sum(torch.log(y_pred_aug1_mix_probas) * pseudo_target_mix, dim=1))
                super_loss = super_loss + mixup_loss
            
            elif self.minimode == 3 or self.minimode == 4:
                y_new = torch.logical_or(y.bool(), self.y_tmp[index].bool()).float()
                super_loss = -torch.mean(torch.sum(torch.log(1.0000001 - y_pred_ori_probas) * (1 - y_new), dim=1))
                
                if epoch >= self.warmup:
                    pi = torch.argmax(y_pred_ori_probas, dim=1)
                    self.y_tmp[index, :] = F.one_hot(pi, y.size()[1]).float()
                
                if self.minimode == 4:
                    # mixup
                    Eta = np.random.beta(8.0, 8.0)
                    idx_rp = torch.randperm(batch_size)
                    x_aug1_mix = Eta * x_aug1 + (1 - Eta) * x_aug1[idx_rp]
                    y_mix = torch.logical_or(y.bool(), y[idx_rp].bool()).float()
                    # y_pred_aug1_mix = model.forward_2(x_aug1_mix)
                    y_pred_aug1_mix =  model(x_ori=x_aug1_mix, eval_only=True)
                    y_pred_aug1_mix_probas = torch.softmax(y_pred_aug1_mix, dim=-1)
                    mixup_loss = -torch.mean(torch.sum(torch.log(1.0000001 - y_pred_aug1_mix_probas) * (1 - y_mix), dim=1))
                    super_loss = super_loss + mixup_loss
            elif self.minimode == 5:
                y_new = torch.logical_or(y.bool(), self.y_tmp[index].bool()).float()
                super_loss = -torch.mean(torch.sum(torch.log(1.0000001 - y_pred_ori_probas) * (1 - y_new), dim=1))
                
                if epoch >= self.warmup:
                    pi = torch.argmax(y_pred_ori_probas, dim=1)
                    self.y_tmp[index, :] = F.one_hot(pi, y.size()[1]).float()
                
                # mixup
                Eta = np.random.beta(8.0, 8.0)
                idx_rp = torch.randperm(batch_size)
                x_aug1_mix = Eta * x_aug1 + (1 - Eta) * x_aug1[idx_rp]
                pseudo_target_tmp = self.pseudo_target[index]
                pseudo_target_mix = Eta * pseudo_target_tmp + (1 - Eta) * pseudo_target_tmp[idx_rp]
                # y_pred_aug1_mix = model.forward_2(x_aug1_mix)
                y_pred_aug1_mix =  model(x_ori=x_aug1_mix, eval_only=True)
                y_pred_aug1_mix_probas = torch.softmax(y_pred_aug1_mix, dim=-1)
                mixup_loss = -torch.mean(torch.sum(torch.log(y_pred_aug1_mix_probas) * pseudo_target_mix, dim=1))
                super_loss = super_loss + mixup_loss

            consist_loss_aug1 = self.loss_InfoNCE(feature_pred_ori, feature_pred_aug1)
            consist_loss_aug2 = self.loss_InfoNCE(feature_pred_ori, feature_pred_aug2)
            # consist_loss_aug1 = 0
            # consist_loss_aug2 = 0
            # consist_loss_aug1 = self.loss_InfoNCE(features=feature_pred_ori, labels=max_pred_indices)
            # consist_loss_aug2 = 0
            # lam = min((epoch / 100) * self.lam, self.lam)

            # 總loss
            final_loss = self.lam * (consist_loss_aug1 + consist_loss_aug2) + super_loss
            # print(final_loss)
            if torch.any(torch.isnan(final_loss)):
                raise ValueError("Log_prob has nan!")
            # training準確度
            y_true =  y_true.cuda()
            final_acc = accuracy(y_pred_ori_probas, y_true)[0]
            # print(y_pred_ori.dtype)
            # print(y_true.dtype)
            CE_Loss = self.loss_CrossEntropy(y_pred_ori, y_true.to(torch.int64))
            # CE_Loss=0.0

            optimizer.zero_grad()
            final_loss.backward()
            optimizer.step()

            if self.minimode == 1 or self.minimode == 2  or self.minimode == 5:
                if epoch >= self.warmup:
                    #更新pseudo_target
                    self.pseudo_target_update(y_pred_ori_probas, index)

            losses.update(final_loss.item(), x_ori.size(0))
            acc.update(final_acc.item())
            CE_Losses.update(CE_Loss.item(), x_ori.size(0))
            # measure elapsed time
            batch_time.update(time.time() - batch_end)
            batch_end = time.time()

            if i % 50 == 0 or i == (len(train_loader)-1):
                logging.info('Epoch: [{0}][{1}/{2}]\t'
                            'Time {3} ({batch_time.avg:.3f})\t'
                            'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                            'CE_Loss {CE_Loss.val:.4f} ({CE_Loss.avg:.4f})\t'
                            'Accuracy {acc.val:.4f} ({acc.avg:.4f})\t'
                            'lam ({lam})\t'.format(
                    epoch, i, len(train_loader), time.time()-end, batch_time=batch_time,
                    data_time=data_time, loss=losses, CE_Loss=CE_Losses, acc=acc, lam=self.lam))
                end = time.time()
        
        return acc.avg, losses.avg
    
    def pseudo_target_update(self, y_pred_ori_probas, index):
        with torch.no_grad():
            if self.minimode == 1 or self.minimode == 5:
                pi = torch.argmax(y_pred_ori_probas, dim=1)
                self.pseudo_target[index, :] = self.pseudo_target[index, :] * self.phi
                self.pseudo_target[index, pi] = self.pseudo_target[index, pi] + (1 - self.phi)
            ####
            elif self.minimode == 2:
                pi = y_pred_ori_probas/y_pred_ori_probas.sum(dim = 1).repeat(y_pred_ori_probas.size()[1], 1).transpose(0, 1)
                self.pseudo_target[index, :] = self.pseudo_target[index, :] * self.phi + pi * (1 - self.phi)

    def validate(self, epoch, test_loader, model):
        batch_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        end = time.time()

        model.eval()
        with torch.no_grad():
            for i, (input, target) in enumerate(test_loader):
                input, target = input.cuda(), target.cuda()
                # target_var = target.cuda()
                
                # compute output
                # output = model.forward_2(input)
                output = model(x_ori=input, eval_only=True)
                # print(torch.argmax(output,dim=1))
                # print(output.dtype)
                # print(target.dtype)
                loss = self.loss_CrossEntropy(output, target)

                output = output.float()
                loss = loss.float()
                # print(output)

                # measure accuracy and record loss
                prec1 = accuracy(output.data, target)[0]
                # print(prec1)
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