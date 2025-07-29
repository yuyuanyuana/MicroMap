import os
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F

from tqdm import tqdm
from MicroMap.model import SpotPriorNet, Token2Expr
from MicroMap.features import UNI_feature
from MicroMap.utils import *

import matplotlib.pyplot as plt
from torch.nn.functional import cosine_similarity

import numpy as np
import pandas as pd
import scanpy as sc
import torch

import pickle
import torch.nn.functional as F



class MicroMap(object):
    def __init__(
        self,
        out_path='.',
        num_workers: int = 4,
        device = 'cuda:0', 
    ):
        super(MicroMap, self).__init__() 
        self.device = device
        self.num_workers = num_workers
        self.out_path = out_path
        os.makedirs(self.out_path, exist_ok=True)
    def cal_spot_prior( self, 
                        spot_count, 
                        n_pca=100, 
                        hidden_dim=64, 
                        latent_dim=32, 
                        batch_size = 64,
                        pretrain_epochs = 200,
                        lambda_kl = 0.01,
                        ):
        # === Spot-level PriorNet 预训练 ===
        self.spot_count = spot_count
        print(f'Preprocessing spot level gene expression with {self.spot_count.shape[0]} spots and {self.spot_count.shape[1]} genes ... ')
        self.spot_pca = prepare_spot_pca(self.spot_count, n_pca=n_pca)
        prior_dataset = TensorDataset(self.spot_pca)
        prior_loader = DataLoader(prior_dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers)
        self.prior_net = SpotPriorNet(input_dim=n_pca, hidden_dim=hidden_dim, latent_dim=latent_dim).to(self.device)
        self.prior_optimizer = torch.optim.Adam(self.prior_net.parameters(), lr=1e-3)
        print(f'Calculating spot specific prior ... ')
        # loss_pre = []
        self.loss_history_prior = {
            'total': [], 'rec': [], 'kl': []
        }
        for epoch in tqdm(range(pretrain_epochs)):
            self.prior_net.train()
            loss_total = 0
            loss_rec = 0
            loss_kl = 0
            for i, (pca_input,) in enumerate(prior_loader):  # 假设 dataloader 现在返回的是 spot PCA 数据
                pca_input = pca_input.to(self.device)  # shape: (B, 100)
                recon_x, mu, logvar = self.prior_net(pca_input)
                # Reconstruction Loss (用 MSE 对 PCA 特征做重建)
                recon_loss = torch.nn.functional.mse_loss(recon_x, pca_input, reduction='mean')
                # KL Loss
                kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
                # Total Loss
                loss_prior = recon_loss + lambda_kl * kl_loss
                self.prior_optimizer.zero_grad()
                loss_prior.backward()
                self.prior_optimizer.step()
                loss_total += loss_prior.item()
                loss_rec += recon_loss.item()
                loss_kl += kl_loss.item()
            self.loss_history_prior['total'].append(loss_total/len(prior_loader))
            self.loss_history_prior['rec'].append(loss_rec/len(prior_loader))
            self.loss_history_prior['kl'].append(loss_kl * lambda_kl/len(prior_loader))
            print(f"[Pretrain Epoch {epoch}] total loss = {loss_total:.4f}")
            # === 每隔 20 个 epoch 画图并保存 ===
            if (epoch != 0) & (epoch % 20 == 0):
                plt.figure(figsize=(10, 6))
                for key in self.loss_history_prior:
                    plt.plot(self.loss_history_prior[key], label=key)
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"Loss at Epoch {epoch}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"loss_prior.png")
                plt.close()
            # loss_pre.append(loss_total)
        torch.save(self.prior_net, f'{self.out_path}/model_prior_net.pt')
    def cal_token_feats( self, 
                         UNI_path = '/data/yyyu/test/UNI/code_raw/assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin', 
                         img_path = None,
                         scale = 0.5, 
                         patch_size = 224,
                         token_size = 16
                         ):
        self.scale = scale
        UNI_feature( model_path = UNI_path,
                     img_path = img_path,
                     scale = scale,
                     patch_size = 224,
                     token_size = 16,
                     stride = 48,
                     out_path = self.out_path,
                     device = self.device)
        file = open(self.out_path + '/features.pickle','rb')
        feats = pickle.load(file)
        self.feats = torch.concat([feats['patch'], feats['token'], feats['rgb']], 2)
    def load_token_feats( self, feats_file=None, feats_value=None, scale=0.5):
        self.scale = scale
        if feats_file is not None:
            feats = pickle.load(open(feats_file,'rb'))
            self.feats = torch.concat([feats['patch'], feats['token'], feats['rgb']], 2)
        if feats_value is not None:
            self.feats = feats_value
    def train( self,         
               spot_count,
               spot_coord, 
               spot_radius,
               genes, 
               hidden_dim = 256, 
               latent_dim = 32,
               dropout = 0.5,
               logvar_scale=1, 
               batch_size = 128, 
               freeze_prior_epochs = 10, 
               lambda_nb = 10, 
               lambda_smooth = 0.01, 
               lambda_kl = 0.001, 
               lambda_kl_prior = 0.01, 
               lambda_prior_recon = 1.0, 
               warmup_epochs = 800, 
               train_epochs = 200,
               weight_smooth_epoch = 100, 
               model_prior_path=None,
               n_pca=100,
               neighbor_type='8'
               ):
        self.genes = genes
        spot_coord = (spot_coord * self.scale).astype(int)
        radius = int(spot_radius * self.scale)
        self.patch_shape = (self.feats.shape[0], self.feats.shape[1])
        self.spot_count = spot_count
        if self.spot_pca is None:
            self.spot_pca = prepare_spot_pca(self.spot_count, n_pca=n_pca)
        dataset = SpotTokenDataset( self.feats, 
                                    self.spot_count, 
                                    self.spot_pca, 
                                    spot_coord, 
                                    radius,
                                    self.patch_shape,
                                    patch_size=16)
        dataloader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=self.num_workers)
        # === 联合训练主干模型 + PriorNet ===
        self.model = Token2Expr( input_dim = self.feats.shape[-1], 
                                 output_dim = self.spot_count.shape[-1],
                                 hidden_dim = hidden_dim, 
                                 latent_dim = latent_dim,
                                 dropout = dropout,
                                 n_batch = 1
                                 ).to(self.device)  
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        if model_prior_path is not None:
            self.prior_net = torch.load(model_prior_path).to(self.device)
            self.prior_optimizer = torch.optim.Adam(self.prior_net.parameters(), lr=1e-3)
        # === 记录每个损失项 ===
        self.loss_history = {
            'total': [], 'nb': [], 'smooth':[], 'size': [],
            'cos_gene': [], 'cos_cell': [],
            'kl_token': [], 'kl_prior': [], 'prior_recon': []
        }
        disk_mask = get_disk_mask(spot_radius//16)
        H, W = disk_mask.shape
        for epoch in tqdm(range(1, train_epochs+1)):
            print(epoch)
            adjust_learning_rate(optimizer, epoch)
            self.model.train()
            self.prior_net.train()
            if epoch <= freeze_prior_epochs:
                for param in self.prior_net.parameters():
                    param.requires_grad = False
            else:
                for param in self.prior_net.parameters():
                    param.requires_grad = True
            loss_tmp = 0
            loss_nb_all = 0
            loss_smooth_all = 0
            loss_size_all = 0
            loss_cos_gene_all = 0
            loss_cos_cell_all = 0
            loss_kl_token_all = 0
            loss_kl_prior_all = 0
            loss_prior_recon_all = 0
            for i, (count_tmp, pca_tmp, size_sum_tmp, feat_tmp) in enumerate(dataloader):
                size_height, size_width = feat_tmp.shape[0], feat_tmp.shape[1]
                pca_tmp = pca_tmp.to(self.device)
                feat_tmp = feat_tmp.to(self.device)
                count_tmp = count_tmp.to(self.device)
                size_sum_tmp = size_sum_tmp.to(self.device)
                # === 1. 构建每个 spot 的先验分布 ===
                rec_pca, mu_prior, logvar_prior = self.prior_net(pca_tmp)
                # === 2. 从 decoder 重建 count_tmp（spot gene expression） ===
                recon_prior = ((rec_pca-pca_tmp)**2).mean()
                # === 3. KL(p(z|x_s) || N(0,1)) ===
                kl_prior = -0.5 * torch.mean(1 + logvar_prior - mu_prior.pow(2) - logvar_prior.exp(), dim=1).mean()
                # === 4. token-level 表达预测 ===
                rate_scaled_tmp, logit_tmp, size_tmp, z_tmp, mu, log_var = self.model(
                                                                                        feat_tmp.flatten(0, 1), 
                                                                                        batch_tensor=torch.zeros(size_height * size_width),
                                                                                        logvar_scale = logvar_scale
                                                                                        )
                rate_tmp = rate_scaled_tmp * size_tmp
                mean_tmp = rate_tmp * logit_tmp
                weights = mean_tmp.view(size_height, size_width, -1).sum(2)
                mean_tmp = mean_tmp.view(size_height, size_width, -1).sum(1)
                rate_tmp = rate_tmp.view(size_height, size_width, -1).sum(1)
                size_tmp = size_tmp.view(size_height, size_width).sum(1)
                # === 4. smooth  ===
                mu_tmp = mu.view(size_height, size_width, -1)
                # z_smooth_loss = fast_smooth_loss_batch(mu_tmp, disk_mask, neighbor_type='24')
                if epoch < weight_smooth_epoch:
                    z_smooth_loss = fast_weighted_smooth_loss_batch(mu_tmp, disk_mask, weights=None, neighbor_type=neighbor_type)
                else:
                    z_smooth_loss = fast_weighted_smooth_loss_batch(mu_tmp, disk_mask, weights=weights, neighbor_type=neighbor_type)
                # z_tmp = z_tmp.view(size_height, H, W, -1).permute(0, 3, 1, 2).contiguous()
                # loss1 = smooth_loss_window(z_tmp, disk_mask)
                # loss2 = laplacian_smooth_loss(z_tmp, disk_mask, neighbor_type='4')
                # z_smooth_loss = loss1 + loss2
                # === 5. token层的多项分布loss + 各种辅助项 ===
                loss_nb_tmp = lambda_nb * nb_loss_func(count_tmp, mean_tmp, rate_tmp).mean()
                loss_size_tmp = 1000 * KL_loss_size(size_sum_tmp, size_tmp)
                cos1 = cosine_similarity(mean_tmp - mean_tmp.mean(), count_tmp - count_tmp.mean(), dim=0).mean()
                cos2 = cosine_similarity(mean_tmp - mean_tmp.mean(), count_tmp - count_tmp.mean(), dim=1).mean()
                loss_cos_gene_tmp = 5 * (1 - cos1.mean())
                loss_cos_cell_tmp = 5 * (1 - cos2.mean())
                # === 6. token latent KL(q(z|x) || p(z|x_s)) ===
                mu_prior_token = mu_prior.unsqueeze(1).expand(-1, size_width, -1).reshape(-1, mu_prior.shape[-1])
                logvar_prior_token = logvar_prior.unsqueeze(1).expand(-1, size_width, -1).reshape(-1, logvar_prior.shape[-1])
                kl_token = kl_normal(mu, log_var, mu_prior_token, logvar_prior_token)
                # kl_token=torch.tensor([0.]).to(self.device)
                # === 7. 总loss ===
                kl_weight = min(epoch / warmup_epochs, 1.0)
                loss_train = (loss_nb_tmp + loss_size_tmp +
                              loss_cos_gene_tmp + loss_cos_cell_tmp +
                              lambda_smooth * z_smooth_loss + 
                              lambda_kl * kl_weight * kl_token +
                              lambda_kl_prior * kl_prior +
                              lambda_prior_recon * recon_prior)
                optimizer.zero_grad()
                if epoch > freeze_prior_epochs:
                    self.prior_optimizer.zero_grad()
                loss_train.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                optimizer.step()
                if epoch > freeze_prior_epochs:
                    torch.nn.utils.clip_grad_norm_(self.prior_net.parameters(), max_norm=5.0)
                    self.prior_optimizer.step()
                loss_tmp += loss_train.item()
                loss_nb_all += loss_nb_tmp.item()
                loss_smooth_all += z_smooth_loss.item()
                loss_size_all += loss_size_tmp.item()
                loss_cos_gene_all += loss_cos_gene_tmp.item()
                loss_cos_cell_all += loss_cos_cell_tmp.item()
                loss_kl_token_all += kl_token.item()
                loss_kl_prior_all += kl_prior.item()
                loss_prior_recon_all += recon_prior.item()
            self.loss_history['total'].append(loss_tmp/len(dataloader))
            self.loss_history['nb'].append(loss_smooth_all/len(dataloader))
            self.loss_history['smooth'].append(loss_nb_all/len(dataloader))
            self.loss_history['size'].append(loss_size_all/len(dataloader))
            self.loss_history['cos_gene'].append(loss_cos_gene_all/len(dataloader))
            self.loss_history['cos_cell'].append(loss_cos_cell_all/len(dataloader))
            self.loss_history['kl_token'].append(loss_kl_token_all/len(dataloader))
            self.loss_history['kl_prior'].append(loss_kl_prior_all/len(dataloader))
            self.loss_history['prior_recon'].append(loss_prior_recon_all/len(dataloader))
            print(f"Epoch {epoch}: total={loss_tmp/len(dataloader):.4f}, nb={loss_nb_all/len(dataloader):.4f}, size={loss_size_all/len(dataloader):.4f}, "
                  f"smooth={loss_smooth_all/len(dataloader):.4f}, "
                  f"cos_gene={loss_cos_gene_all/len(dataloader):.4f}, cos_cell={loss_cos_cell_all/len(dataloader):.4f}, kl_token={loss_kl_token_all/len(dataloader):.4f}, "
                  f"kl_prior={loss_kl_prior_all/len(dataloader):.4f}, prior_recon={loss_prior_recon_all/len(dataloader):.4f}")
            # === 每隔 20 个 epoch 画图并保存 ===
            if (epoch != 0) & (epoch % 20 == 0):
                plt.figure(figsize=(10, 6))
                for key in self.loss_history:
                    plt.plot(self.loss_history[key], label=key)
                plt.xlabel("Epoch")
                plt.ylabel("Loss")
                plt.title(f"Loss at Epoch {epoch}")
                plt.legend()
                plt.tight_layout()
                plt.savefig(f"loss.png")
                plt.close()
            if (epoch != 0) & (epoch % 50 == 0):
                torch.save(self.model, f'{self.out_path}/model_{epoch}.pt')
    def predict(self, 
                model_path = None, 
                genes = None, 
                add_img=False, 
                img_path=None, 
                index_use=None, 
                extract_mask=True,
                batch_size=12800,
                logvar_scale=1): 
        self.patch_shape = (self.feats.shape[0], self.feats.shape[1])
        patch_all = np.vstack(
                               (np.arange(self.patch_shape[0]).repeat(self.patch_shape[1]), 
                                np.tile(np.arange(self.patch_shape[1]), self.patch_shape[0])
                               )
                             ).T
        index_all = [str(ii) for ii in list(pd.DataFrame(patch_all).apply(tuple, axis=1).values)]
        dataloader_all = DataLoader(self.feats.flatten(0,1), shuffle=False, batch_size=batch_size, num_workers=self.num_workers)
        if model_path is not None:
            self.model = torch.load(model_path)
        self.model.eval()
        means = []
        zs = []
        for i, feat_tmp in enumerate(dataloader_all):
            size_use = feat_tmp.size(0)
            feat_tmp = feat_tmp.to(self.device)
            rate_scaled_tmp, logit_tmp, size_tmp, z_tmp, _, _ = self.model(feat_tmp, batch_tensor=torch.zeros(size_use), logvar_scale=logvar_scale, infer_mode=True)
            rate_tmp = rate_scaled_tmp * size_tmp
            mean_tmp = rate_tmp * logit_tmp
            means.append(mean_tmp.detach().cpu())
            zs.append(z_tmp.detach().cpu())
        expr_hir = torch.cat(means).numpy()
        late_hir = torch.cat(zs).numpy()
        print(expr_hir.shape)
        expr_hir = pd.DataFrame(expr_hir, index=index_all, columns=genes)
        adata_hir = sc.AnnData(expr_hir) 
        adata_hir.obs[['patch_y', 'patch_x']] = patch_all
        adata_hir.obsm['spatial'] = (adata_hir.obs[['patch_y', 'patch_x']] * (16/self.scale) + (8/self.scale)).values
        adata_hir.obsm['latent'] = late_hir
        if add_img:
            image = cv2.imread(img_path)
            spatial_key = "spatial"
            library_id = "fullres"
            adata_hir.uns[spatial_key] = {library_id: {}}
            adata_hir.uns[spatial_key][library_id]["images"] = {}
            adata_hir.uns[spatial_key][library_id]["images"] = {"hires": image}
            adata_hir.uns[spatial_key][library_id]["scalefactors"] = {
                "tissue_hires_scalef": 1,
                "spot_diameter_fullres": 16/self.scale,
            }
        if extract_mask:
            if index_use is None:
                mask = get_mask(self.feats) 
                patch_mask = np.vstack(np.where(mask)).T
                index_mask = [str(ii) for ii in list(pd.DataFrame(patch_mask).apply(tuple, axis=1).values)]
                index_use = batch_query_indices(index_mask, index_all)
            adata_mask = adata_hir[index_use].copy()
            if add_img:
                adata_mask = adata_mask[(adata_mask.obsm['spatial'][:,0] < image.shape[0])&(adata_mask.obsm['spatial'][:,1] < image.shape[1])].copy()
            return adata_hir, adata_mask, index_use
        else:
            return  adata_hir



