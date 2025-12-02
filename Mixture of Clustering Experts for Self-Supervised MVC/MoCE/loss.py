import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class Loss(nn.Module):
    def __init__(self, batch_size, class_num, temperature_l, device):
        super(Loss, self).__init__()
        self.batch_size = batch_size
        self.class_num = class_num
        self.temperature_l = temperature_l
        self.device = device

        self.mask = self.mask_correlated_samples(batch_size)
        self.similarity = nn.CosineSimilarity(dim=2)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.KL_function = nn.KLDivLoss(reduction='batchmean')
        self.Mse_funcion = nn.MSELoss()

    def mask_correlated_samples(self, N):
        mask = torch.ones((N, N))
        mask = mask.fill_diagonal_(0)
        for i in range(N//2):
            mask[i, N//2 + i] = 0
            mask[N//2 + i, i] = 0
        mask = mask.bool()
        return mask


    def forward_label(self, q_i, q_j):
        p_i = q_i.sum(0).view(-1)
        p_i /= p_i.sum()
        ne_i = math.log(p_i.size(0)) + (p_i * torch.log(p_i)).sum()
        p_j = q_j.sum(0).view(-1)
        p_j /= p_j.sum()
        ne_j = math.log(p_j.size(0)) + (p_j * torch.log(p_j)).sum()
        entropy = ne_i + ne_j

        q_i = q_i.t()
        q_j = q_j.t()
        N = 2 * self.class_num
        q = torch.cat((q_i, q_j), dim=0)

        sim = self.similarity(q.unsqueeze(1), q.unsqueeze(0)) / self.temperature_l
        sim_i_j = torch.diag(sim, self.class_num)
        sim_j_i = torch.diag(sim, -self.class_num)

        positive_clusters = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        mask = self.mask_correlated_samples(N)
        negative_clusters = sim[mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_clusters.device).long()
        logits = torch.cat((positive_clusters, negative_clusters), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss + entropy

    def CS_divergence(self, p, q):
        p = torch.clamp(p, min=1e-10)  # 防止除以零
        q = torch.clamp(q, min=1e-10)

        cs_divergence = 1 - (torch.sqrt(p) * torch.sqrt(q)).sum() ** 2 / (p.sum() * q.sum())
        return cs_divergence

    def KL_loss(self, p, q):

        loss = self.KL_function(q.log(), p)
        return loss

    def MSE_loss(self, x, x_bar):

        loss = self.Mse_funcion(x, x_bar)

        return loss

    def diver_loss(self, q_exp):
        M = q_exp.size(1)
        loss = 0.0
        count = 0
        for i in range(M):
            for j in range(i+1, M):
                qi = q_exp[:, i, :].mean(dim=0)
                qj = q_exp[:, j, :].mean(dim=0)
                sim = F.cosine_similarity(qi, qj, dim=0)
                loss += sim
                count += 1

        return loss / count
