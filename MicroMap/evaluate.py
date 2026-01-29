from scipy.stats import pearsonr, spearmanr
from skimage.metrics import structural_similarity as ssim
# from skimage.metrics import peak_signal_noise_ratio as psnr
# from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
from scipy.stats import rankdata 

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


def pre_count(cnts_raw):
    row_sums = cnts_raw.sum(axis=1)
    conversion_factors = 1000 / (row_sums + 1e-10)  # 避免除以0
    conversion_factors = pd.Series(conversion_factors, index=cnts_raw.index)
    normalized_array = cnts_raw.mul(conversion_factors, axis=0)
    log_transformed_array = np.log1p(normalized_array)
    return log_transformed_array



from joblib import Parallel, delayed
from skimage.metrics import structural_similarity as ssim
from scipy.stats import rankdata

# ===============================
# 1. PCC & RMSE
# ===============================
def eval_pcc_rmse_streaming(pred_mm, gt_mm, genes_use=None, batch_size=20000, dtype=np.float32):
    if isinstance(pred_mm, pd.DataFrame):
        if genes_use is not None:
            pred = pred_mm.loc[:, genes_use].to_numpy(dtype=dtype, copy=False)
            gt   = gt_mm.loc[:,  genes_use].to_numpy(dtype=dtype, copy=False)
            gene_index = list(genes_use)
        else:
            pred = pred_mm.to_numpy(dtype=dtype, copy=False)
            gt   = gt_mm.to_numpy(dtype=dtype, copy=False)
            gene_index = list(pred_mm.columns)
    else:
        pred = np.asarray(pred_mm, dtype=dtype)
        gt   = np.asarray(gt_mm,   dtype=dtype)
        gene_index = np.arange(pred.shape[1])
    n_cells, n_genes = pred.shape
    sumA = np.zeros(n_genes, dtype=np.float64)
    sumB = np.zeros(n_genes, dtype=np.float64)
    sumA2 = np.zeros(n_genes, dtype=np.float64)
    sumB2 = np.zeros(n_genes, dtype=np.float64)
    sumAB = np.zeros(n_genes, dtype=np.float64)
    sse   = np.zeros(n_genes, dtype=np.float64)
    n_seen = 0
    for i in range(0, n_cells, batch_size):
        j = min(i + batch_size, n_cells)
        A = pred[i:j, :]
        B = gt[i:j,   :]
        sumA  += A.sum(axis=0, dtype=np.float64)
        sumB  += B.sum(axis=0, dtype=np.float64)
        sumA2 += (A*A).sum(axis=0, dtype=np.float64)
        sumB2 += (B*B).sum(axis=0, dtype=np.float64)
        sumAB += (A*B).sum(axis=0, dtype=np.float64)
        D = A - B
        sse  += (D*D).sum(axis=0, dtype=np.float64)
        n_seen += (j - i)
    n = float(n_seen)
    varA = (sumA2 - (sumA * sumA) / n) / max(n - 1.0, 1.0)
    varB = (sumB2 - (sumB * sumB) / n) / max(n - 1.0, 1.0)
    covAB = (sumAB - (sumA * sumB) / n) / max(n - 1.0, 1.0)
    denom = np.sqrt(varA * varB)
    denom[denom == 0] = np.nan
    pcc = covAB / denom
    pcc = np.nan_to_num(pcc, nan=0.0)
    rmse = np.sqrt(sse / n)
    return pd.DataFrame({"PCC": pcc.astype(np.float32), "RMSE": rmse.astype(np.float32)}, index=gene_index)


# ===============================
# 2. Spearman
# ===============================
def _spearman_one_gene(x, y):
    rx = rankdata(x, method='average')
    ry = rankdata(y, method='average')
    rx = (rx - rx.mean()) / rx.std(ddof=1)
    ry = (ry - ry.mean()) / ry.std(ddof=1)
    return float(np.dot(rx, ry) / (rx.size - 1))

def spearman_by_column(A, B, n_jobs=8):
    G = A.shape[1]
    vals = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(_spearman_one_gene)(A[:, j], B[:, j]) for j in range(G)
    )
    return np.array(vals, dtype=np.float32)


# ===============================
# 3. SSIM
# ===============================
def ssim_spatial_all(pred, gt, genes_use, index_gt, patch_shape,
                     win_size=11, n_jobs=8, chunk_size=256, gaussian_weights=True):
    H, W = patch_shape
    if isinstance(pred, pd.DataFrame):
        A = pred.loc[:, genes_use].to_numpy(copy=False)
        B = gt.loc[:,   genes_use].to_numpy(copy=False)
        gene_index = list(genes_use)
    else:
        A = np.asarray(pred); B = np.asarray(gt)
        gene_index = list(genes_use)
    # print("A NaN:", np.isnan(A).any(), "A Inf:", np.isinf(A).any())
    # print("B NaN:", np.isnan(B).any(), "B Inf:", np.isinf(B).any())
    def _chunk(genes_slice):
        data1 = np.zeros((H, W), dtype=np.float32)
        data2 = np.zeros((H, W), dtype=np.float32)
        vals = []
        for j in genes_slice:
            data1.fill(0.0); data2.fill(0.0)
            data1[index_gt] = A[:, j]
            data2[index_gt] = B[:, j]
            v1 = data1[index_gt]; v2 = data2[index_gt]
            dr = float(max(v1.max(), v2.max()) - min(v1.min(), v2.min()) + 1e-12)
            s = ssim(data1, data2, channel_axis=None, win_size=win_size,
                     gaussian_weights=gaussian_weights, data_range=1)
            vals.append(s)
        return vals
    G = len(gene_index)
    slices = [range(i, min(i+chunk_size, G)) for i in range(0, G, chunk_size)]
    results = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_chunk)(sl) for sl in slices)
    ss = [v for part in results for v in part]
    return pd.Series(ss, index=gene_index, dtype=np.float32)


# ===============================
# 4. evaluation_func
# ===============================
def evaluation_func(pred_mm, gt_mm, genes_use,
                    do_spearman=True, spearman_subsample=100000, spearman_jobs=8,
                    do_ssim=True, ssim_kwargs=None):
    # 1) PCC + RMSE
    res = eval_pcc_rmse_streaming(pred_mm, gt_mm, genes_use=genes_use, batch_size=20000)
    # 2) Spearman
    if do_spearman:
        if isinstance(pred_mm, pd.DataFrame):
            n = pred_mm.shape[0]
            idx = np.random.choice(n, min(spearman_subsample, n), replace=False)
            A = pred_mm.iloc[idx, :].loc[:, genes_use].to_numpy(copy=False)
            B = gt_mm.iloc[idx,   :].loc[:, genes_use].to_numpy(copy=False)
        else:
            n = pred_mm.shape[0]
            idx = np.random.choice(n, min(spearman_subsample, n), replace=False)
            A = pred_mm[idx, :]
            B = gt_mm[idx,   :]
        rcc = spearman_by_column(A, B, n_jobs=spearman_jobs)
        res["RCC"] = rcc
    # 3) SSIM
    if do_ssim:
        if ssim_kwargs is None or "index_gt" not in ssim_kwargs or "patch_shape" not in ssim_kwargs:
            raise ValueError("SSIM need ssim_kwargs 'index_gt' and 'patch_shape'")
        ssim_series = ssim_spatial_all(pred_mm, gt_mm, genes_use=genes_use,
                                       index_gt=ssim_kwargs["index_gt"],
                                       patch_shape=ssim_kwargs["patch_shape"],
                                       win_size=ssim_kwargs.get("win_size", 11),
                                       n_jobs=ssim_kwargs.get("n_jobs", 8),
                                       chunk_size=ssim_kwargs.get("chunk_size", 256),
                                       gaussian_weights=ssim_kwargs.get("gaussian_weights", True))
        res["SSIM"] = ssim_series.astype(np.float32)
    # print(res.mean(numeric_only=True))
    return res




