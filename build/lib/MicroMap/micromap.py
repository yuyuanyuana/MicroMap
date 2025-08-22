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

    """
    MicroMap: A deep generative framework for predicting high-resolution 
    spatial transcriptomic profiles directly from H&E-stained histology images.
    
    """

    def __init__(
        self,
        out_path='.',
        num_workers: int = 4,
        device = 'cuda:0', 
    ):
        """
        Initialize MicroMap.

        Args:
            out_path (str): Output directory for saving results. Default is current directory ('.').
            num_workers (int): Number of worker processes for data loading. Default is 4.
            device (str): Device to run computations on (e.g., 'cuda:0' or 'cpu'). Default is 'cuda:0'.
        """

        super(MicroMap, self).__init__() 
        self.device = device
        self.num_workers = num_workers
        self.out_path = out_path
        os.makedirs(self.out_path, exist_ok=True)
    

    def cal_spot_prior( self, 
                        spot_count, 
                        n_pca=100, 
                        hidden_dim=64, 
                        latent_dim=64, 
                        batch_size = 64,
                        pretrain_epochs = 200,
                        lambda_kl = 0.01,
                        ):
        
        """
        Compute the prior distribution of gene expression latent space for each spot.

        Args:
            spot_count (ndarray or tensor): Gene expression count matrix at spot level.
            n_pca (int): Number of principal components used for dimensionality reduction. Default is 100.
            hidden_dim (int): Dimension of the hidden layer in the VAE encoder and decoder. Default is 64.
            latent_dim (int): Dimension of the latent space for gene expression representation. Default is 32.
            batch_size (int): Mini-batch size for training. Default is 64.
            pretrain_epochs (int): Number of epochs for pretraining. Default is 200.
            lambda_kl (float): Weight of KL divergence in the loss function. Default is 0.01.

        Returns:
            dict: Learned spot-level latent prior parameters (e.g., mean and variance).
        """

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
            
            for i, (pca_input,) in enumerate(prior_loader):  
                
                pca_input = pca_input.to(self.device) 
                recon_x, mu, logvar = self.prior_net(pca_input)
                # Reconstruction Loss 
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
            # # === 每隔 20 个 epoch 画图并保存 ===
            # if (epoch != 0) & (epoch % 20 == 0):
            #     plt.figure(figsize=(10, 6))
            #     for key in self.loss_history_prior:
            #         plt.plot(self.loss_history_prior[key], label=key)
            #     plt.xlabel("Epoch")
            #     plt.ylabel("Loss")
            #     plt.title(f"Loss at Epoch {epoch}")
            #     plt.legend()
            #     plt.tight_layout()
            #     plt.savefig(f"loss_prior.png")
            #     plt.close()
            # loss_pre.append(loss_total)
        # torch.save(self.prior_net, f'{self.out_path}/model_prior_net.pt')
    
    
    def cal_token_feats( self, 
                         UNI_path = '/data/yyyu/test/UNI/code_raw/assets/ckpts/vit_large_patch16_224.dinov2.uni_mass100k/pytorch_model.bin', 
                         img_path = None,
                         scale = 1.0,  
                         patch_size = 224,
                         token_size = 16
                         ):
        
        """
        Extract token-level features from histology image using a pretrained UNI model.

        Args:
            UNI_path (str): Path to the pretrained UNI model checkpoint. 
                            Default is vit_large_patch16_224 pretrained on 100k images.
            img_path (str): Path to the input histology image. Required.
            scale (float): Down-sampling ratio applied before feature extraction. Default is 1.0.
            patch_size (int): Size of image patches fed into the model. Default is 224.
            token_size (int): Token granularity within each patch. Default is 16.

        Returns:
            torch.Tensor: Concatenated feature matrix of shape [N, M, D].
        """

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
    

    def load_token_feats( self, feats_file=None, feats_value=None, scale=1):

        """
        Load pre-computed token-level image features, either from a file or from memory.

        Args:
            feats_file (str, optional): Path to a pickle file containing features.
                                        The file should store a dict with keys:
                                        ['patch', 'token', 'rgb'].
            feats_value (torch.Tensor, optional): Directly provide a feature tensor 
                                                  (e.g. pre-loaded or pre-processed).
            scale (float): Down-sampling ratio of the corresponding image. 
                           Must match the scale used during feature extraction.

        Notes:
            - If both feats_file and feats_value are provided, feats_value has priority.
            - Final features are concatenated along the last dimension. 
        """

        self.scale = scale
        
        if feats_file is not None:
            
            feats = pickle.load(open(feats_file,'rb'))
            self.feats = torch.concat([feats['patch'], feats['token'], feats['rgb']], 2)
        
        else:
            
            self.feats = feats_value


    def train( self,         
               spot_count,
               spot_coord, 
               spot_radius,
               genes, 
               hidden_dim = 256, 
               latent_dim = 64,
               dropout = 0.5,
               logvar_scale=0.1, 
               batch_size = 128, 
               freeze_prior_epochs = 210, 
               lambda_nb = 1, 
               lambda_smooth = 0.01, 
               lambda_kl = 0.001, 
               lambda_kl_prior = 0.0001, 
               lambda_prior_recon = 0.01, 
               warmup_epochs = 800, 
               train_epochs = 200,
               weight_smooth_epoch = 100, 
               model_prior_path=None,
               n_pca=100,
               neighbor_type='8'
               ):

        """
        Train the main model for predicting spatial gene expression from histology image features.

        Args:
            spot_count (ndarray or tensor): Gene expression counts for each spot.
            spot_coord (ndarray): XY-coordinates of spots in image space.
            spot_radius (float): Radius of each spot (in pixels).
            genes (list): List of gene names corresponding to spot_count.
            
            hidden_dim (int): Hidden dimension of Token2Expr encoder. 
            latent_dim (int): Latent space dimension. 
            dropout (float): Dropout rate in Token2Expr. 
            logvar_scale (float): Scaling factor for log variance in VAE. 
            batch_size (int): Batch size for training. 
            freeze_prior_epochs (int): Number of epochs to freeze PriorNet. 
            
            lambda_nb (float): Weight for negative binomial loss. 
            lambda_smooth (float): Weight for spatial smoothness loss. 
            lambda_kl (float): Weight for latent KL loss (token-level). 
            lambda_kl_prior (float): Weight for KL loss of PriorNet.
            lambda_prior_recon (float): Weight for PriorNet reconstruction. 
            
            warmup_epochs (int): Number of epochs to linearly increase KL weight. Default 800.
            train_epochs (int): Total number of training epochs. Default 200.
            weight_smooth_epoch (int): Epoch after which smoothness loss is weighted by expression. 
            
            model_prior_path (str, optional): Path to pretrained PriorNet model checkpoint.
            n_pca (int): Number of PCA components used for spot count preprocessing. 
            neighbor_type (str): Type of neighborhood connectivity for smoothness loss 
                                 (e.g., '4', '8', '24'). 
        Returns:
            None
            (Trained models are stored in self.model and self.prior_net; 
             training history in self.loss_history; checkpoints/loss plots saved to out_path.)
        """

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
        

        self.model = Token2Expr( input_dim = self.feats.shape[-1], 
                                 output_dim = self.spot_count.shape[-1],
                                 hidden_dim = hidden_dim, 
                                 latent_dim = latent_dim,
                                 dropout = dropout,
                                 n_batch = 1
                                 ).to(self.device)  
        
        ptimizer = torch.optim.Adam(self.model.parameters(), lr=1e-2)
        
        if model_prior_path is not None:
            self.prior_net = torch.load(model_prior_path).to(self.device)
            self.prior_optimizer = torch.optim.Adam(self.prior_net.parameters(), lr=1e-3)
        

        self.loss_history = {
            'total': [], 'nb': [], 'smooth':[], 'size': [],
            'cos_gene': [], 'cos_cell': [],
            'kl_token': [], 'kl_prior': [], 'prior_recon': []
        }
        
        disk_mask = get_disk_mask(spot_radius//16)
        
        H, W = disk_mask.shape
        
        for epoch in tqdm(range(1, train_epochs+1)):
            
            # print(epoch)
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
                
                # === 1. ===
                rec_pca, mu_prior, logvar_prior = self.prior_net(pca_tmp)
                # === 2. ===
                recon_prior = ((rec_pca-pca_tmp)**2).mean()
                # === 3. ===
                kl_prior = -0.5 * torch.mean(1 + logvar_prior - mu_prior.pow(2) - logvar_prior.exp(), dim=1).mean()
                # === 4. ===
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
                
                # === 5. ===
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
                # === 6. ===
                loss_nb_tmp = lambda_nb * nb_loss_func(count_tmp, mean_tmp, rate_tmp).mean()
                loss_size_tmp = 1000 * KL_loss_size(size_sum_tmp, size_tmp)
                cos1 = cosine_similarity(mean_tmp - mean_tmp.mean(), count_tmp - count_tmp.mean(), dim=0).mean()
                cos2 = cosine_similarity(mean_tmp - mean_tmp.mean(), count_tmp - count_tmp.mean(), dim=1).mean()
                loss_cos_gene_tmp = 5 * (1 - cos1.mean())
                loss_cos_cell_tmp = 5 * (1 - cos2.mean())
                # === 7. ===
                mu_prior_token = mu_prior.unsqueeze(1).expand(-1, size_width, -1).reshape(-1, mu_prior.shape[-1])
                logvar_prior_token = logvar_prior.unsqueeze(1).expand(-1, size_width, -1).reshape(-1, logvar_prior.shape[-1])
                kl_token = kl_normal(mu, log_var, mu_prior_token, logvar_prior_token)
                # kl_token=torch.tensor([0.]).to(self.device)
                # === 8. ===
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
            # # === 每隔 20 个 epoch 画图并保存 ===
            # if (epoch != 0) & (epoch % 20 == 0):
            #     plt.figure(figsize=(10, 6))
            #     for key in self.loss_history:
            #         plt.plot(self.loss_history[key], label=key)
            #     plt.xlabel("Epoch")
            #     plt.ylabel("Loss")
            #     plt.title(f"Loss at Epoch {epoch}")
            #     plt.legend()
            #     plt.tight_layout()
            #     plt.savefig(f"loss.png")
            #     plt.close()
            # if (epoch != 0) & (epoch % 50 == 0):
            #     torch.save(self.model, f'{self.out_path}/model_{epoch}.pt')


    def predict(self, 
                model_path = None, 
                genes = None, 
                add_img=False, 
                img_path=None, 
                index_use=None, 
                extract_mask=True,
                batch_size=12800): 
    
        """
        Predict high-resolution gene expression from image features using the trained model.

        This function performs token-level prediction by passing the flattened image features 
        through the model, reconstructing mean expression values and latent representations 
        for each spatial patch. It can optionally add the corresponding histological image and 
        extract a mask of relevant patches.

        Parameters
        ----------
        model_path : str or None, optional
            Path to a pre-trained model to load. If None, the current model instance is used.
        genes : list of str
            List of gene names corresponding to the output dimensions of the model.
        add_img : bool, default False
            If True, reads the high-resolution image from `img_path` and stores it in AnnData.
        img_path : str or None
            Path to the high-resolution histology image. Required if `add_img` is True.
        index_use : list of int or None, optional
            Subset of patch indices to use when extracting the mask. If None, the mask is generated automatically.
        extract_mask : bool, default True
            If True, returns an AnnData object filtered by the tissue mask, along with the mask indices.
        batch_size : int, default 12800
            Number of patches to process per batch during prediction.
        logvar_scale : float, default 1
            Scaling factor for the model's log-variance during inference.

        Returns
        -------
        adata_hir : AnnData
            Full-resolution predicted gene expression with spatial coordinates and latent embeddings.
        adata_mask : AnnData, optional
            Masked subset of `adata_hir` corresponding to tissue regions. Returned if `extract_mask=True`.
        index_use : list of int, optional
            Indices of the patches used in the masked AnnData. Returned if `extract_mask=True`.

        Notes
        -----
        - The function flattens the feature map and processes it in batches for memory efficiency.
        - Latent representations (`latent`) are stored in `adata_hir.obsm['latent']`.
        - Spatial coordinates are adjusted based on the model scale.
        - The optional image is stored in `adata_hir.uns['spatial']['fullres']['images']['hires']`.
        
        """

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
            
            rate_scaled_tmp, logit_tmp, size_tmp, z_tmp, _, _ = self.model( feat_tmp, 
                                                                            batch_tensor=torch.zeros(size_use), 
                                                                            logvar_scale=0.1, 
                                                                            infer_mode=True)
            
            rate_tmp = rate_scaled_tmp * size_tmp
            mean_tmp = rate_tmp * logit_tmp
            means.append(mean_tmp.detach().cpu())
            zs.append(z_tmp.detach().cpu())
        
        expr_hir = torch.cat(means).numpy()
        late_hir = torch.cat(zs).numpy()
        
        # print(expr_hir.shape)
        
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



