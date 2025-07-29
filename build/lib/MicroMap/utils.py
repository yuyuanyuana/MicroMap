import numpy as np
import torch 
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ============================================================================ #
# utils

def batch_query_indices(small_list, large_list):
    index_map = {value: index for index, value in enumerate(large_list)}
    indices = [index_map.get(element, -1) for element in small_list] 
    return indices 


def normalize_count(cnts_raw):
    cnts = cnts_raw.copy()
    cnts_min = cnts.min(0)
    cnts_max = cnts.max(0)
    cnts -= cnts_min
    cnts /= (cnts_max - cnts_min) + 1e-12
    return cnts


def scale_count(cnts_raw):
    cnts = cnts_raw.copy()
    cnts_mean = cnts.mean(0)
    cnts_std = cnts.std(0)
    cnts -= cnts_mean
    cnts /= cnts_std + 1e-12
    return cnts



# ============================================================================ #
# get tissue mask from embedding 

from scipy.ndimage import binary_erosion, binary_dilation
import numpy as np
import cv2

def correct_mask(mask, num_patches=10):
    eroded_mask = binary_erosion(mask, structure=np.ones((3, 3)), iterations=1)
    edge_mask = mask & ~eroded_mask
    expanded_edge_mask = binary_dilation(edge_mask, structure=np.ones((3, 3)), iterations=num_patches)
    corrected_mask = mask & ~expanded_edge_mask
    return corrected_mask


def auto_invert_center(labels):
    rows, cols = labels.shape
    start_row, end_row = rows//4, 3*rows//4
    start_col, end_col = cols//4, 3*cols//4
    center_region = labels[start_row:end_row, start_col:end_col]
    count_ones = np.sum(center_region)
    total_pixels = center_region.size
    if count_ones < total_pixels / 2:
        return 1 - labels
    else:
        return labels.copy()


def get_mask(embs, margin=16, num_patches=2, path='.'):
    from sklearn.decomposition import PCA
    shapes = (embs.shape[0], embs.shape[1])
    data = embs.flatten(0,1).numpy()
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    data_standardized = (data - data_mean) / data_std
    n_components = 10
    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_standardized)
    data_use = data_pca
    from sklearn.cluster import KMeans
    model_km = KMeans(n_clusters=2,random_state=0, verbose=0)
    labels = model_km.fit_predict(data_use)
    labels_use = auto_invert_center(labels.reshape(shapes[0],shapes[1]))
    # sum_corner = labels_use[0,0] + labels_use[-1,0] + labels_use[0,-1] + labels_use[-1,-1]
    # if sum_corner > 2:
    #     labels_use = 1-labels_use
    if margin != 0:
        labels_use[:margin, :] = 0
        labels_use[-margin:, :] = 0
        labels_use[:, :margin] = 0
        labels_use[:, -margin:] = 0
    mask = labels_use * 255
    cv2.imwrite(f'{path}/mask0.png', mask)
    mask = mask.astype(bool)
    mask = correct_mask(mask, num_patches=num_patches)
    return mask



# ============================================================================ #
# spot_pca

def prepare_spot_pca(cnts, n_pca=100):
    """
    对 spot-level 表达矩阵进行预处理，并做 PCA 降维。
    Args:
        cnts (np.ndarray or torch.Tensor): 输入 count 矩阵，shape = [n_spot, n_gene]
        device (torch.device): 返回的 tensor 目标设备 (如 'cuda' or 'cpu')
        n_pca (int): PCA 降维维度数，默认100

    Returns:
        torch.Tensor: PCA 后的特征，shape = [n_spot, n_pca]
    """
    # 如果输入是 torch tensor，先转 numpy
    if isinstance(cnts, torch.Tensor):
        cnts_np = cnts.cpu().numpy()
    else:
        cnts_np = cnts
    # Step 1: CPM-like normalization + log1p
    counts_per_million = cnts_np / (cnts_np.sum(axis=1, keepdims=True) + 1e-8) * 1e5
    log_counts = np.log1p(counts_per_million)
    # Step 2: Standardize across genes (zero mean, unit variance per gene)
    scaler = StandardScaler()
    log_counts_scaled = scaler.fit_transform(log_counts)
    # Step 3: PCA 降维
    pca = PCA(n_components=n_pca, random_state=42)
    pca_feats = pca.fit_transform(log_counts_scaled)
    # Step 4: 转 torch tensor
    # pca_tensor = torch.tensor(pca_feats, dtype=torch.float32).to(device)
    return torch.tensor(pca_feats, dtype=torch.float32)



# ============================================================================ #
# learning_rate

def adjust_learning_rate(optimizer, epoch):
    if epoch <= 100:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-2
    elif 100 < epoch <= 450:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-3
    else:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 1e-4



# ============================================================================ #
# loss

import torch.nn.functional as F
def KL_loss_size(size_sum_tmp, size_tmp):
    # 归一化函数
    def normalize_size(vector):
        return vector / vector.sum()
    # 归一化两个向量
    P = normalize_size(size_sum_tmp)
    Q = normalize_size(size_tmp)
    # 计算 KL 散度
    return F.kl_div(Q.log(), P, reduction='batchmean')


def nb_loss_func(data, mean, disp):
    eps = 1e-10
    loss1 = torch.lgamma(disp+eps) + torch.lgamma(data+1) - torch.lgamma(data+disp+eps)
    loss2 = (disp+data) * torch.log(1.0 + (mean/(disp+eps))) + (data * (torch.log(disp+eps) - torch.log(mean+eps)))
    return loss1 + loss2


def kl_normal(mu_q, logvar_q, mu_p, logvar_p):
    # Ensure numerical stability
    logvar_q = torch.clamp(logvar_q, min=-10, max=10)
    logvar_p = torch.clamp(logvar_p, min=-10, max=10)
    # Compute variances
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (
        logvar_p - logvar_q +
        (var_q + (mu_q - mu_p) ** 2) / var_p - 1
    )
    return kl.mean()


def kl_standard_normal(mu_q, logvar_q, mu_p, logvar_p):
    # Ensure numerical stability
    logvar_q = torch.clamp(logvar_q, min=-10, max=10)
    logvar_p = torch.clamp(logvar_p, min=-10, max=10)
    # Compute variances
    var_q = torch.exp(logvar_q)
    var_p = torch.exp(logvar_p)
    kl = 0.5 * (
        logvar_p - logvar_q +
        (var_q + (mu_q - mu_p) ** 2) / var_p - 1
    )
    return kl.mean()



def fast_weighted_smooth_loss_batch(z_spot_batch, disk_mask, weights=None, neighbor_type='4'):
    """
    Args:
        z_spot_batch: [N, T=49, D] 表示 N 个 spot 的 token latent 表达
        disk_mask: (H, W)，bool numpy array，表示 token 的空间排列
        weights: [N, T]，每个 token 的权重（可选），用于加权平滑惩罚项
        neighbor_type: '4', '8', or '24'
    Returns:
        scalar smoothness loss
    """
    N, T, D = z_spot_batch.shape
    device = z_spot_batch.device
    H, W = disk_mask.shape
    # Step 1: 构建 [N, D, H, W] 的 token 映射
    z_map_batch = torch.zeros(N, D, H, W, device=device)
    idx = torch.tensor(np.argwhere(disk_mask), device=device)  # [T, 2]
    for t in range(T):
        i, j = idx[t]
        z_map_batch[:, :, i, j] = z_spot_batch[:, t, :]
    # Step 2: 卷积核 K，模拟离散拉普拉斯
    if neighbor_type == '4':
        kernel = torch.tensor([[0, -1, 0],
                               [-1, 4, -1],
                               [0, -1, 0]], dtype=torch.float32, device=device)
        pad = 1
    elif neighbor_type == '8':
        kernel = torch.tensor([[-1, -1, -1],
                               [-1,  8, -1],
                               [-1, -1, -1]], dtype=torch.float32, device=device)
        pad = 1
    elif neighbor_type == '24':
        kernel = torch.tensor([[-1, -1, -1, -1, -1],
                               [-1, -1, -1, -1, -1], 
                               [-1, -1, 24, -1, -1],
                               [-1, -1, -1, -1, -1],
                               [-1, -1, -1, -1, -1]], dtype=torch.float32, device=device)
        pad = 2
    else:
        raise ValueError("neighbor_type must be '4', '8', or '24'")
    kernel = kernel.view(1, 1, *kernel.shape).repeat(D, 1, 1, 1)  # [D, 1, K, K]
    # Step 3: 卷积得到 Laplacian 响应
    z_lap = F.conv2d(z_map_batch, kernel, padding=pad, groups=D)  # [N, D, H, W]
    loss_map = z_lap ** 2  # [N, D, H, W]
    # Step 4: Mask
    mask = torch.from_numpy(disk_mask).to(device).float()  # [H, W]
    loss_masked = loss_map * mask.view(1, 1, H, W)  # [N, D, H, W]
    # Step 5: 加权平均（可选）
    if weights is not None:
        # weights: [N, T]
        weights = weights + 1e-8
        weights = weights / weights.sum(dim=1, keepdim=True)  # Normalize to sum=1 per spot
        weight_map = torch.zeros(N, 1, H, W, device=device)
        for t in range(T):
            i, j = idx[t]
            weight_map[:, 0, i, j] = weights[:, t]  # 填入每个位置的权重
        loss_weighted = (loss_masked * weight_map).sum() / (N * D)
        return loss_weighted
    else:
        return loss_masked.sum() / (mask.sum() * D * N + 1e-8)


# ============================================================================ #
# dataset

class SpotTokenDataset(Dataset):
    def __init__(self, embs1, 
                       expr1, 
                       pca1, 
                       locs1, 
                       radius1, 
                       patch_shape1,
                       patch_size=16):
        super().__init__()
        mask1 = get_disk_mask(radius1//patch_size)
        locs1 = locs1//patch_size
        feat_dim = embs1.shape[-1]
        embs1 = embs1.reshape(patch_shape1[0], patch_shape1[1], feat_dim)
        self.x1 = get_patches_flat(embs1, locs1, mask1)
        self.expr1 = expr1
        self.pca1 = pca1
        self.size = expr1.sum(1)
    def __len__(self):
        return self.x1.shape[0]
    def __getitem__(self, idx):
        return self.expr1[idx], self.pca1[idx], self.size[idx], self.x1[idx]


def get_patches_flat(img, locs, mask):
    shape = np.array(mask.shape)
    center = shape // 2
    r = np.stack([-center, shape-center], -1)  # offset
    x_list = []
    for s in locs:
        print(s)
        patch = img[ s[0]+r[0][0]:s[0]+r[0][1],
                     s[1]+r[1][0]:s[1]+r[1][1] ]
        x = patch[mask,:]
        x_list.append(x)
    x_list = np.stack(x_list)
    return x_list


def get_disk_mask(radius, boundary_width=None):
    radius_ceil = np.ceil(radius).astype(int)
    locs = np.meshgrid(
            np.arange(-radius_ceil, radius_ceil+1),
            np.arange(-radius_ceil, radius_ceil+1),
            indexing='ij')
    locs = np.stack(locs, -1)
    distsq = (locs**2).sum(-1)
    isin = distsq <= radius**2
    if boundary_width is not None:
        isin *= distsq >= (radius-boundary_width)**2
    return isin


def get_disk(img, ij, radius):
    i, j = ij
    patch = img[i-radius:i+radius, j-radius:j+radius]
    disk_mask = get_disk_mask(radius)
    patch[~disk_mask] = 0.0
    return patch


def get_masked_patches(locs_use, radius, patch_size, patch_shape):
    patch_mask = get_disk_mask(radius//patch_size)
    shape = np.array(patch_mask.shape)
    center = shape // 2
    r = np.stack([-center, shape-center], -1)  # offset
    patch_masked_list = []
    for s in locs_use:
        print(s)
        img_patch = np.zeros(patch_shape)
        img_patch[ s[0]+r[0][0]:s[0]+r[0][1], s[1]+r[1][0]:s[1]+r[1][1] ] = patch_mask.astype(int)
        patch_masked_tmp = np.vstack(np.where(img_patch)).T
        patch_masked_list.append(patch_masked_tmp)
    patches_masked = np.vstack(patch_masked_list)
    return patches_masked

