import torch
from torch import nn
import torch.nn.functional as F

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)): # 計算準確度(Testing時)
    """Computes the precision@k for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# class InfoNCE(nn.Module): # InfoNCE loss
#     def __init__(self, temperature=1.0):
#         super().__init__()
#         self.temperature = temperature

#     def forward(self, z_i, z_j):
#         batch_size = z_i.size(0)
#         z = torch.cat([z_i, z_j], dim=0)
#         similarity = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=2)

#         sim_ij = torch.diag(similarity, batch_size)
#         sim_ji = torch.diag(similarity, -batch_size)
#         positives = torch.cat([sim_ij, sim_ji], dim=0)

#         mask = (~torch.eye(batch_size * 2, batch_size * 2, dtype=torch.bool)).float().cuda()
#         numerator = torch.exp(positives / self.temperature)
#         denominator = mask * torch.exp(similarity / self.temperature)

#         all_losses = -torch.log(numerator / torch.sum(denominator, dim=1))
#         loss = torch.sum(all_losses) / (2 * batch_size)
#         return loss
    
# class SupConLoss(nn.Module): # PiCO loss
#     """Following Supervised Contrastive Learning: 
#         https://arxiv.org/pdf/2004.11362.pdf."""
#     def __init__(self, temperature=0.07, base_temperature=0.07):
#         super().__init__()
#         self.temperature = temperature
#         self.base_temperature = base_temperature

#     def forward(self, features, mask=None, batch_size=-1):
#         device = (torch.device('cuda')
#                   if features.is_cuda
#                   else torch.device('cpu'))

#         if mask is not None:
#             # SupCon loss (Partial Label Mode)
#             mask = mask.float().detach().to(device)
#             # compute logits
#             anchor_dot_contrast = torch.div(
#                 torch.matmul(features[:batch_size], features.T),
#                 self.temperature)
#             # for numerical stability
#             logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
#             logits = anchor_dot_contrast - logits_max.detach()

#             # mask-out self-contrast cases
#             logits_mask = torch.scatter(
#                 torch.ones_like(mask),
#                 1,
#                 torch.arange(batch_size).view(-1, 1).to(device),
#                 0
#             )
#             mask = mask * logits_mask

#             # compute log_prob
#             exp_logits = torch.exp(logits) * logits_mask
#             log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
#             # compute mean of log-likelihood over positive
#             mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

#             # loss
#             loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
#             loss = loss.mean()
#         else:
#             # MoCo loss (unsupervised)
#             # compute logits
#             # Einstein sum is more intuitive
#             # positive logits: Nx1
#             q = features[:batch_size]
#             k = features[batch_size:batch_size*2]
#             queue = features[batch_size*2:]
#             l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
#             # negative logits: NxK
#             l_neg = torch.einsum('nc,kc->nk', [q, queue])
#             # logits: Nx(1+K)
#             logits = torch.cat([l_pos, l_neg], dim=1)

#             # apply temperature
#             logits /= self.temperature

#             # labels: positive key indicators
#             labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
#             loss = F.cross_entropy(logits, labels)

#         return loss
    
# class ConLoss(nn.Module): # ABLE loss
#     def __init__(self, base_temperature=0.07):
#         super().__init__()
#         self.temperature = 0.1
#         self.base_temperature = base_temperature
    
#     def forward(self, confidence, outputs, features, Y, index):
#         batch_size = Y.size()[0]
#         device = torch.device('cuda')
        
#         contrast_count = features.shape[1]
#         contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        
#         anchor_feature = contrast_feature
#         anchor_count = contrast_count
        
#         anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        
#         logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True) 
        
#         logits = anchor_dot_contrast - logits_max.detach() 
                
#         Y = Y.float()
        
#         output_sm = F.softmax(outputs[0 : batch_size, :], dim=1).float()
#         output_sm_d = output_sm.detach()
#         _, target_predict = (output_sm_d * Y).max(1)

#         predict_labels =  target_predict.repeat(batch_size, 1).to(device)
        
#         mask_logits = torch.zeros_like(predict_labels).float().to(device)
        
#         pos_set = (Y == 1.0).nonzero().to(device)
        
#         ones_flag = torch.ones(batch_size).float().to(device)
#         zeros_flag = torch.zeros(batch_size).float().to(device)
        
#         for pos_set_i in range(pos_set.shape[0]):
#             sample_idx = pos_set[pos_set_i][0]
#             class_idx = pos_set[pos_set_i][1]
#             mask_logits_tmp = torch.where(predict_labels[sample_idx] == class_idx, ones_flag, zeros_flag).float()
#             if mask_logits_tmp.sum() > 0:
#                 mask_logits_tmp = mask_logits_tmp / mask_logits_tmp.sum()
#                 mask_logits[sample_idx] = mask_logits[sample_idx] + mask_logits_tmp * confidence[sample_idx][class_idx]

#         mask_logits = mask_logits.repeat(anchor_count, contrast_count)
        
#         logits_mask = torch.scatter(
#             torch.ones_like(mask_logits),
#             1,
#             torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
#             0
#         ).float()

#         mask_logits = mask_logits * logits_mask
#         exp_logits = logits_mask * torch.exp(logits)
#         log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
#         mean_log_prob_pos = (mask_logits * log_prob).sum(1)
        
#         loss_con_m = - (self.temperature / self.base_temperature) * mean_log_prob_pos
#         loss_con = loss_con_m.view(anchor_count, batch_size).mean()
        
#         revisedY_raw = Y.clone()
#         revisedY_raw = revisedY_raw * output_sm_d
#         revisedY_raw = revisedY_raw / revisedY_raw.sum(dim = 1).repeat(Y.size()[1], 1).transpose(0, 1)
#         new_target = revisedY_raw.detach()
        
#         return loss_con, new_target