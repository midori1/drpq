r'''
calculation of pairwise distance, and return condensed result, i.e. we omit the diagonal and duplicate entries and store everything in a one-dimensional array
'''
import torch
import numpy as np
import time

def forgy(X, valid_mask, n_clusters, random_select=True):
    batch_size, len_x, hdim = X.size()
    max_sample_loc = valid_mask.sum(-1).min(0).values.long().detach().cpu().numpy()
    indices = np.zeros((batch_size, n_clusters), dtype=np.int32)
    for b in range(batch_size):
        if random_select:
            if max_sample_loc < n_clusters:
                indices[b] = np.random.choice(max_sample_loc, size=(n_clusters,), replace=True)
            else:
                indices[b] = np.random.choice(max_sample_loc, size=(n_clusters,), replace=False)
        else:
            # remove randomness
            cur_max_loc = valid_mask[b].sum().long().detach().cpu().numpy()
            inds = [(cur_max_loc // n_clusters)*i for i in range(n_clusters)]
            indices[b] = [1 if ind == 0 else ind for ind in inds]  # skip [CLS]
    indices = torch.tensor(indices).long().to(X.device).unsqueeze(-1).expand(-1, -1, hdim)  # [batch, n_clusters, hdim]
    initial_state = X.gather(1, indices)  # [batch, n_clusters, hdim]
    return initial_state


def lloyd(X, valid_mask, n_clusters, tol=1e-4, max_iter=20, random_select=True):
    initial_state = forgy(X, valid_mask, n_clusters, random_select)  # [batch_size, n_clusters, hdim]
    iter = 0
    while iter < max_iter:
        dis = pairwise_distance(X, initial_state)  # dis [batch_size, len1, n_clusters]

        choice_cluster = torch.argmin(dis, dim=-1)  # [batch_size, len1]

        initial_state_pre = initial_state.clone()

        # update each cluster center
        for index in range(n_clusters):
            selected_mask = (choice_cluster == index).float()  # [batch_size, len1]
            selected_mask = selected_mask * valid_mask.float()
            num_selected = selected_mask.sum(1) + 1e-7  # [batch_size] prevent nan
            initial_state[:, index, :] = (X * selected_mask.unsqueeze(-1)).sum(1) / num_selected.unsqueeze(-1)

        center_shift = torch.sum(torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1)))
        if center_shift ** 2 < tol:
            break
        iter += 1

    return choice_cluster, initial_state


def lloyd_timer(X, valid_mask, n_clusters, tol=1e-4, max_iter=20, random_select=True):
    time_start = time.time()
    initial_state = forgy(X, valid_mask, n_clusters, random_select)  # [batch_size, n_clusters, hdim]
    iter = 0
    while iter < max_iter:
        dis = pairwise_distance(X, initial_state)  # dis [batch_size, len1, n_clusters]

        choice_cluster = torch.argmin(dis, dim=-1)  # [batch_size, len1]

        initial_state_pre = initial_state.clone()

        # update each cluster center
        for index in range(n_clusters):
            selected_mask = (choice_cluster == index).float()  # [batch_size, len1]
            selected_mask = selected_mask * valid_mask.float()
            num_selected = selected_mask.sum(1) + 1e-7  # [batch_size] prevent nan
            initial_state[:, index, :] = (X * selected_mask.unsqueeze(-1)).sum(1) / num_selected.unsqueeze(-1)

        center_shift = torch.sum(torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1)))
        if center_shift ** 2 < tol:
            break
        iter += 1
    time_end = time.time()
    time_cost = time_end - time_start

    return choice_cluster, initial_state, iter, time_cost



def pairwise_distance(data1, data2=None):
    if data2 is None:
        data2 = data1

    # dis [batch_size, len1, 1]
    dis1 = torch.sum(data1 ** 2, dim=-1, keepdim=True)
    # dis2 [batch_size, len1, len2]
    dis2 = -2 * torch.matmul(data1, data2.transpose(1, 2))
    # dis3 [batch_size, 1, len2]
    dis3 = torch.sum(data2 ** 2, dim=-1, keepdim=True).transpose(1, 2)
    # dis [batch_size, len1, len2]
    dis = dis1 + dis2 + dis3
    return dis


def pairwise_dot(data1, data2=None):
    if data2 is None:
        data2 = data1

    # [batch_size, len1, len2]
    dis = torch.bmm(data1, data2.transpose(1, 2))
    dis = torch.nn.functional.softmax(dis, dim=-1)
    return dis


def group_pairwise(X, groups, device=0, fun=lambda r, c: pairwise_distance(r, c).cpu()):
    group_dict = {}
    for group_index_r, group_r in enumerate(groups):
        for group_index_c, group_c in enumerate(groups):
            R, C = X[group_r], X[group_c]
            if device != -1:
                R = R.cuda(device)
                C = C.cuda(device)
            group_dict[(group_index_r, group_index_c)] = fun(R, C)
    return group_dict
