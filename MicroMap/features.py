from PIL import Image
import numpy as np
from torchvision import transforms
import torch
import timm
# from huggingface_hub import login, hf_hub_download
from PIL import Image
Image.MAX_IMAGE_PIXELS = 10000000000
from skimage.transform import rescale
import re
from einops import rearrange, repeat, reduce
import pickle
import os


transform = transforms.Compose([ transforms.ToTensor(),
                                 transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                                ])


def pad_img(image, patch_size=224):
    
    shape_ori = np.array(image.shape[:2])
    shape_ext = ( ((shape_ori + patch_size - 1) // patch_size)*patch_size )
    
    image = np.pad( image,
                    ((0, shape_ext[0] - image.shape[0]),
                    (0, shape_ext[1] - image.shape[1]),
                    (0, 0)),
                mode='edge')
    
    return image


def batch_transform(image, batch_h=3, batch_w=3):
    
    h, w, c = image.shape
    block_height = int(np.ceil( h / batch_h))  
    block_width = int(np.ceil( w / batch_w )) 
    starts_h = list(range(0, h, block_height))
    starts_w = list(range(0, w, block_width))
    blocks = [[] for _ in range(len(starts_h))]
    
    for i in range(len(starts_h)):
        
        for j in range(len(starts_w)):
            
            start_h = starts_h[i]
            start_w = starts_w[j]
            
            end_h = start_h + block_height
            end_w = start_w + block_width
            
            block = image[start_h:end_h, start_w:end_w, :]
            block = transform(block/255)
            blocks[i].append(block)
    
    blocks_r = [torch.concat(block_, 2) for block_ in blocks]
    out = torch.concat(blocks_r, 1)
    
    return out


def split_patches(image, patch_size=224):
    
    height, width, channels = image.shape
    num_patches_height = height // patch_size
    num_patches_width = width // patch_size
    
    patches = image.reshape(num_patches_height, patch_size, num_patches_width, patch_size, channels)
    patches = patches.swapaxes(1, 2).reshape(-1, patch_size, patch_size, channels)
    
    patches_tensor = torch.from_numpy(patches).permute(0, 3, 1, 2) 
    
    return patches_tensor.float(), image, num_patches_height, num_patches_width


def load_image(filename, verbose=True):
    
    img = Image.open(filename)
    
    return np.array(img)


def UNI_feature(model_path = '.',
                patch_size = 224,
                token_size = 16,
                UNI2 = False, 
                scale = 0.5,
                stride = 64,
                batch_size = 2560, 
                img_path = '.',
                out_path = '.',
                device = 'cuda:1',
                return_result = True):
    
    img = load_image(img_path)
    img = rescale(img, [scale, scale, 1], preserve_range=True)
    # print('Image shape before padding: ', img.shape)
    img = pad_img(img, patch_size=224)
    # print('Image shape after padding: ', img.shape)
    os.makedirs(out_path, mode=0o777, exist_ok=True)
    Image.fromarray(img.astype(np.uint8)).save(f'{out_path}/img_processed.jpg')
    # print('Image saved! ')
    # all_numbers = re.findall(r'\d+', model_weight_name)
    # patch_size = int(all_numbers[-1])
    # token_size = int(all_numbers[-2])
    print('Creating and loading model ... ')
    if not UNI2:
        model = timm.create_model(
            "vit_large_patch16_224", img_size=224, patch_size=16, init_values=1e-5, num_classes=0, dynamic_img_size=True
        )
    else:
        timm_kwargs = {
                'model_name': 'vit_giant_patch14_224',
                'img_size': 224, 
                'patch_size': 14, 
                'depth': 24,
                'num_heads': 24,
                'init_values': 1e-5, 
                'embed_dim': 1536,
                'mlp_ratio': 2.66667*2,
                'num_classes': 0, 
                'no_embed_class': True,
                'mlp_layer': timm.layers.SwiGLUPacked, 
                'act_layer': torch.nn.SiLU, 
                'reg_tokens': 8, 
                'dynamic_img_size': True
            }
        model = timm.create_model(pretrained=False, **timm_kwargs)
        token_size = 14
        patch_size = 224
    model.load_state_dict(torch.load(model_path), strict=True)
    # print('Put model on device ... ')
    model = model.to(device)
    model.eval()
    print('Extracting features... ')
    input_tensor = torch.randn(1, 3, patch_size, patch_size)
    output_tensor = model.forward_features(input_tensor.to(device))
    feature_dim = output_tensor.shape[2]
    shape_emb = np.array(img.shape[:2]) // token_size
    features_shift_token =  torch.zeros((shape_emb[0], shape_emb[1], feature_dim)).float()
    features_shift_patch =  torch.zeros((shape_emb[0], shape_emb[1], feature_dim)).float()
    # print('Shifting ... ')
    for start_shift_h in range(0, patch_size, stride):
        for start_shift_w in range(0, patch_size, stride):
            # print('start_shift_h: ', start_shift_h, 'start_shift_w: ', start_shift_w)
            img_tmp = img[start_shift_h:, start_shift_w:, :]
            img_tmp = np.pad( img_tmp, ((0, start_shift_h), (0, start_shift_w), (0, 0)), mode='edge')
            img_transformed = batch_transform(img_tmp, batch_h=5, batch_w=3)
            patches, image, num_patches_height, num_patches_width = split_patches(img_transformed.permute(1,2,0).numpy(), patch_size=patch_size)
            sizes = patches.shape[0]
            # batch_size = 2560
            # print('shape of patches: ', patches.shape)
            # print('shape of patches: ', patches.shape)
            features = []
            for start in range(0, sizes, batch_size):
                end = start + batch_size
                with torch.inference_mode():
                    feature_emb = model.forward_features(patches[start:end,:,:,:].to(device)) # Extracted features (torch.Tensor) with shape [1,1024]
                features.append(feature_emb.cpu())
            features = torch.concat(features, 0)
            # print('shape of features: ', features.shape)
            features_patch = features[:,0,:]
            features_token = features[:,1:,:]
            features_token = rearrange( features_token, '(h1 w1) (h2 w2) k -> (h1 h2) (w1 w2) k',
                                            h1=num_patches_height, w1=num_patches_width, h2=int(patch_size//token_size), w2=int(patch_size//token_size))
            features_patch = rearrange( features_patch, '(h1 w1) k -> h1 w1 k',
                                            h1=num_patches_height, w1=num_patches_width)
            features_patch = repeat(  features_patch, 'h12 w12 k -> (h12 h3) (w12 w3) k',
                                            h3=int(patch_size//token_size), w3=int(patch_size//token_size))
            start_token_h = start_shift_h//token_size
            start_token_w = start_shift_w//token_size
            end_token_h = shape_emb[0] - start_token_h
            end_token_w = shape_emb[1] - start_token_w
            features_shift_token[start_token_h:, start_token_w:, :] += features_token[:end_token_h, :end_token_w, :] 
            features_shift_patch[start_token_h:, start_token_w:, :] += features_patch[:end_token_h, :end_token_w, :] 
    features_shift_token = features_shift_token / (int(np.ceil(patch_size / stride)) ** 2)
    features_shift_patch = features_shift_patch / (int(np.ceil(patch_size / stride)) ** 2)
    embs = dict(patch=features_shift_patch, token=features_shift_token)
    rgb = torch.from_numpy(np.stack([
            reduce(
                img[..., i].astype(np.float16) / 255.0,
                '(h1 h) (w1 w) -> h1 w1', 'mean',
                h=token_size, w=token_size).astype(np.float32)
            for i in range(3)]).transpose(1,2,0))   
    feat = torch.cat(
        [features_shift_patch, features_shift_token, rgb],
        dim=2
    )
    print('Saving features ...')
    os.makedirs(out_path, exist_ok=True)
    save_path = f"{out_path}/features.pt"
    torch.save(feat, save_path)
    print("Saved to:", save_path)
    if return_result:
        return feat


