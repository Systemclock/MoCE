import math

import numpy as np
import torch
from typing import Optional
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score, accuracy_score
from sklearn.metrics.cluster._supervised import check_clusterings
from scipy.optimize import linear_sum_assignment
from torch.utils.data import Dataset
import torch.nn.functional as F
# nmi ari acc pur fscore

def evaluate(label, pred):

    nmi = normalized_mutual_info_score(label, pred)
    ari = adjusted_rand_score(label, pred)
    resment, acc = cluster_accuracy(label, pred)
    pur = purity(label, pred)
    pre, recal, fscore = b3_precision_recall_fscore(label, pred)
    return acc, nmi, ari, pur, fscore

def diver_loss(q_exp):
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

def purity(y_true, y_pred):
    y_voted_labels = np.zeros(y_true.shape)
    labels = np.unique(y_true)
    ordered_labels = np.arange(labels.shape[0])
    for k in range(labels.shape[0]):
        y_true[y_true == labels[k]] = ordered_labels[k]
    labels = np.unique(y_true)
    bins = np.concatenate((labels, [np.max(labels)+1]), axis=0)

    for cluster in np.unique(y_pred):
        hist, _ = np.histogram(y_true[y_pred == cluster], bins=bins)
        winner = np.argmax(hist)
        y_voted_labels[y_pred == cluster] = winner

    return accuracy_score(y_true, y_voted_labels)


def cluster_accuracy(y_true, y_predicted, cluster_number: Optional[int] = None):
    """
    Calculate clustering accuracy after using the linear_sum_assignment function in SciPy to
    determine reassignments.

    :param y_true: list of true cluster numbers, an integer array 0-indexed
    :param y_predicted: list  of predicted cluster numbers, an integer array 0-indexed
    :param cluster_number: number of clusters, if None then calculated from input
    :return: reassignment dictionary, clustering accuracy
    """
    if cluster_number is None:
        cluster_number = (
            max(y_predicted.max(), y_true.max()) + 1
        )  # assume labels are 0-indexed
    count_matrix = np.zeros((cluster_number, cluster_number), dtype=np.int64)
    for i in range(y_predicted.size):
        count_matrix[y_predicted[i], y_true[i]] += 1

    row_ind, col_ind = linear_sum_assignment(count_matrix.max() - count_matrix)
    reassignment = dict(zip(row_ind, col_ind))
    accuracy = count_matrix[row_ind, col_ind].sum() / y_predicted.size
    return reassignment, accuracy

def b3_precision_recall_fscore(labels_true, labels_pred):
    """Compute the B^3 variant of precision, recall and F-score.
    Parameters
    ----------
    :param labels_true: 1d array containing the ground truth cluster labels.
    :param labels_pred: 1d array containing the predicted cluster labels.
    Returns
    -------
    :return float precision: calculated precision
    :return float recall: calculated recall
    :return float f_score: calculated f_score
    Reference
    ---------
    Amigo, Enrique, et al. "A comparison of extrinsic clustering evaluation
    metrics based on formal constraints." Information retrieval 12.4
    (2009): 461-486.
    """
    # Check that labels_* are 1d arrays and have the same size

    labels_true, labels_pred = check_clusterings(labels_true, labels_pred)

    # Check that input given is not the empty set
    if labels_true.shape == (0,):
        raise ValueError(
            "input labels must not be empty.")

    # Compute P/R/F scores
    n_samples = len(labels_true)
    true_clusters = {}  # true cluster_id => set of sample indices
    pred_clusters = {}  # pred cluster_id => set of sample indices

    for i in range(n_samples):
        true_cluster_id = labels_true[i]
        pred_cluster_id = labels_pred[i]

        if true_cluster_id not in true_clusters:
            true_clusters[true_cluster_id] = set()
        if pred_cluster_id not in pred_clusters:
            pred_clusters[pred_cluster_id] = set()

        true_clusters[true_cluster_id].add(i)
        pred_clusters[pred_cluster_id].add(i)

    for cluster_id, cluster in true_clusters.items():
        true_clusters[cluster_id] = frozenset(cluster)
    for cluster_id, cluster in pred_clusters.items():
        pred_clusters[cluster_id] = frozenset(cluster)

    precision = 0.0
    recall = 0.0

    intersections = {}

    for i in range(n_samples):
        pred_cluster_i = pred_clusters[labels_pred[i]]
        true_cluster_i = true_clusters[labels_true[i]]

        if (pred_cluster_i, true_cluster_i) in intersections:
            intersection = intersections[(pred_cluster_i, true_cluster_i)]
        else:
            intersection = pred_cluster_i.intersection(true_cluster_i)
            intersections[(pred_cluster_i, true_cluster_i)] = intersection

        precision += len(intersection) / len(pred_cluster_i)
        recall += len(intersection) / len(true_cluster_i)

    precision /= n_samples
    recall /= n_samples

    f_score = 2 * precision * recall / (precision + recall)

    return precision, recall, f_score

def target_distribution(batch: torch.Tensor) -> torch.Tensor:

    weight = (batch ** 2) / torch.sum(batch, 0)
    return (weight.t() / torch.sum(weight, 1)).t()


def new_P(inputs, centers):
    # eq (11)
    alpha = 1
    q = 1.0 / (1.0 + (np.sum(np.square(np.expand_dims(inputs, axis=1) - centers), axis=2) / alpha))
    q **= (alpha + 1.0) / 2.0
    q = np.transpose(np.transpose(q) / np.sum(q, axis=1))
    return torch.tensor(q)

def cross_view_consistency_loss(cross_z_list):
    num_views = len(cross_z_list)
    loss = 0.
    count = 0
    for i in range(num_views):
        for j in range(i + 1, num_views):
            loss += F.mse_loss(cross_z_list[i], cross_z_list[j])
            count += 1

    if count > 0:
        loss = loss / count
    return loss

def orthogonal_loss(prototypes):
    """
    prototypes: Tensor of shape [C, D], where C is number of clusters / experts.
    """
    proto_norm = F.normalize(prototypes, dim=1)  # [C, D]

    # Compute similarity matrix: M M^T
    # sim_matrix = torch.matmul(proto_norm, proto_norm.T)  # [C, C]
    # off_diag_mask = ~torch.eye(sim_matrix.size(0), dtype=torch.bool, device=sim_matrix.device)
    # off_diag_values = sim_matrix[off_diag_mask]  # [C*(C-1)]
    # loss = torch.mean(off_diag_values ** 2)  # closer to 0

    # consine similarity
    sim = torch.mm(proto_norm, proto_norm.T)
    off_diag = sim - torch.diag(torch.diag(sim))  # set diag=0
    loss = torch.mean(off_diag ** 2)    # = (1/(C(C-1))) ¦²_{i¡Ùj} cos2(i,j)


    return loss

def gaussian_kernel_matrix(H, sigma=1.0):

    B = H.size(0)
    H_square = (H ** 2).sum(dim=1, keepdim=True)  # [B, 1]
    dist_sq = H_square + H_square.T - 2 * H @ H.T  # [B, B]

    K = torch.exp(-dist_sq / (2 * sigma ** 2))     # ¸ßË¹ºË
    return K
# ?????
def gating_orthogonality_loss(gate_probs):
    """
    gate_probs: Tensor of shape [B, M], softmaxºóÃ¿¸öÑù±¾µÄ×¨¼ÒÑ¡Ôñ¸ÅÂÊ
    """
    # ±ê×¼»¯Ã¿¸öÑù±¾µÄÃÅ¿ØÏòÁ¿
    gate_norm = F.normalize(gate_probs, p=2, dim=1)  # [B, M]

    # ÏàËÆ¶È¾ØÕó£¨Gram£©
    G = torch.matmul(gate_norm, gate_norm.T)  # [B, B]

    # È¥³ý¶Ô½ÇÏß£¬±£Áô·Ç¶Ô½ÇÏî
    off_diag = G - torch.eye(G.size(0), device=G.device)

    # Ê¹ÓÃ Frobenius ·¶ÊýÔ¼Êø·Ç¶Ô½ÇÏîÎª 0£¨Ô½Ð¡Ô½Õý½»£©
    loss = torch.norm(off_diag, p='fro') ** 2 / (G.size(0) * (G.size(0) - 1))
    return loss



def compute_repel_loss(mu_stack, margin=1.0):
    """
    Vectorized repel loss across experts for each view.
    mu_stack: List[Tensor]£¬Ã¿¸öÔªËØÊÇ [E, C, D]
    """
    loss = 0.0
    view = len(mu_stack)

    for mu in mu_stack:  # mu: [E, C, D]
        E, C, D = mu.shape
        if E < 2:
            continue  # ÎÞ·¨¼ÆËã pairwise loss

        # ¼ÆËã×¨¼ÒÖÐÐÄÖ®¼äµÄÁ½Á½¾àÀë£¨ÅÅ³ý×Ô¶Ô£©
        mu_i = mu.unsqueeze(1)        # [E, 1, C, D]
        mu_j = mu.unsqueeze(0)        # [1, E, C, D]
        dists = torch.norm(mu_i - mu_j, dim=-1)  # [E, E, C]

        # ÉÏÈý½Ç mask£¨ÅÅ³ý×Ô¼ººÍÖØ¸´×éºÏ£©
        mask = torch.triu(torch.ones((E, E), device=mu.device), diagonal=1).unsqueeze(-1)  # [E, E, 1]
        dists = dists * mask  # Ö»±£ÁôÉÏÈý½Ç

        # ¼ÆËã³Í·£Ïî
        repel = F.relu(margin - dists).pow(2)  # [E, E, C]

        valid_pairs = (mask > 0).sum().item() * C
        if valid_pairs > 0:
            loss += repel.sum() / valid_pairs

    return loss / view if view > 0 else torch.tensor(0.0, device=mu_stack[0].device)

def orthogonality_loss(mu_stack):
    """
    ¶ÔÃ¿¸ö×¨¼ÒµÄ¾ÛÀàÖÐÐÄ¾ØÕó mu[e] ¡Ê [C, D] Ìí¼ÓÕý½»Ô¼Êø
    mu_stack: List[Tensor]£¬Ã¿¸ö Tensor ÊÇ [E, C, D]
    """
    loss = 0.0
    count = 0
    for mu in mu_stack:  # shape: [E, C, D]
        E, C, D = mu.shape
        for e in range(E):
            centers = mu[e]  # shape: [C, D]

            # Normalize across D
            centers_norm = F.normalize(centers, p=2, dim=1)  # shape: [C, D]

            # Cosine similarity matrix: [C, C]
            cos_sim = torch.matmul(centers_norm, centers_norm.T)  # [C, C]

            identity = torch.eye(C, device=mu.device)
            loss += ((cos_sim - identity) ** 2).mean()
            count += 1

    return loss / count if count > 0 else torch.tensor(0.0, device=mu_stack[0].device)