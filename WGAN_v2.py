import os
import glob
import argparse
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, utils as vutils
from PIL import Image
from tqdm import tqdm
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.stats import gaussian_kde, norm
import warnings
from plots import plot_evaluation_curves, plot_score_distributions, plot_patch_heatmap
from cfg import cfg, CFG
warnings.filterwarnings('ignore')

os.makedirs(cfg.out_dir, exist_ok=True)
os.makedirs(cfg.ckpt_dir, exist_ok=True)

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(42)

class MVTecLOCO_PatchDataset(Dataset):
    def __init__(self, root: str, category: str, mode: str='train', patch_size:int=128, stride:int=64,
                 image_resize:int=512, transform=None, precompute:bool=True, max_patches:int=20000):
        self.root = Path(root)
        self.category = category
        self.mode = mode
        self.patch_size = patch_size
        self.stride = stride
        self.image_resize = image_resize
        self.transform = transform
        self.precompute = precompute
        self.max_patches = max_patches

        if mode == 'train':
            folder = self.root / category / "train" / "good"
        elif mode == 'train1':
            folder = self.root / category / "train" / "good"
        elif mode == 'val':
            folder = self.root / category / "validation" / "good"
        elif mode == 'test':
            folder = self.root / category / "test"  
        else:
            raise ValueError("mode must be train/val/test")

        if not folder.exists():
            raise FileNotFoundError(f"{folder} not found")

        exts = ('*.png','*.jpg','*.jpeg')
        files = []
        for e in exts:
            files.extend(sorted(glob.glob(str(folder / e))))
        
        if len(files) == 0:
            raise FileNotFoundError(f"No images in {folder}")

        self.files = files

        if transform is None:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
            ])
        else:
            self.transform = transform

        self.patches = []
        if self.precompute and mode == 'train':
            self._precompute_patches()

    def _precompute_patches(self):
        patches = []
        for p in tqdm(self.files, desc="Precomputing patches"):
            try:
                img = Image.open(p).convert("RGB")
                img = img.resize((self.image_resize, self.image_resize))
                W, H = img.size
                for top in range(0, H - self.patch_size + 1, self.stride):
                    for left in range(0, W - self.patch_size + 1, self.stride):
                        patch = img.crop((left, top, left+self.patch_size, top+self.patch_size))
                        if self.transform:
                            patch = self.transform(patch)
                        patches.append(patch)
                        if len(patches) >= self.max_patches:
                            break
                    if len(patches) >= self.max_patches:
                        break
            except Exception as e:
                print(f"Skipping {p}: {e}")
            if len(patches) >= self.max_patches:
                break
        random.shuffle(patches)
        self.patches = patches[:self.max_patches]
        print(f"Precomputed {len(self.patches)} patches from {len(self.files)} images")

    def __len__(self):
        if self.mode == 'train' and self.precompute:
            return len(self.patches)
        else:
            return len(self.files)

    def __getitem__(self, idx):
        if self.mode == 'train' and self.precompute:
            return self.patches[idx], 0
        elif self.mode in ('val','test'):
            p = self.files[idx]
            img = Image.open(p).convert("RGB").resize((self.image_resize, self.image_resize))
            tensor = self.transform(img)
            label = 0 if 'good' in str(p) else 1
            return tensor, p, label
        else:
            p = random.choice(self.files)
            img = Image.open(p).convert("RGB").resize((self.image_resize, self.image_resize))
            W, H = img.size
            left = random.randint(0, W - self.patch_size)
            top = random.randint(0, H - self.patch_size)
            patch = img.crop((left, top, left + self.patch_size, top + self.patch_size))
            patch_t = self.transform(patch)
            return patch_t, 0

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1 or classname.find('InstanceNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            nn.init.constant_(m.bias.data, 0.0)

class Generator(nn.Module):
    def __init__(self, nz=128, ngf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz, ngf * 16, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 16),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 16, ngf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z):
        return self.main(z)

class Discriminator(nn.Module):
    def __init__(self, nc=3, ndf=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 16, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ndf * 16, affine=True),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.conv6 = nn.Sequential(
            nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=True)
        )

    def forward(self, x):
        f1 = self.conv1(x)
        f2 = self.conv2(f1)
        f3 = self.conv3(f2)
        f4 = self.conv4(f3)
        f5 = self.conv5(f4)
        out = self.conv6(f5)
        return out.view(out.size(0)), [f1, f2, f3, f4, f5]

class Encoder(nn.Module):
    def __init__(self, nz=128, ngf=64, nc=3):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(nc, ngf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ngf * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ngf * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ngf * 4, ngf * 8, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ngf * 8, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(ngf * 8, ngf * 16, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(ngf * 16, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(ngf * 16 * 4 * 4, nz)
        self.tanh = nn.Tanh()

    def forward(self, x):
        features = self.main(x)
        features_flat = features.view(features.size(0), -1)
        z = self.fc(features_flat)
        z = self.tanh(z)
        return z

def compute_gradient_penalty(D: nn.Module, real_samples: torch.Tensor, fake_samples: torch.Tensor, device: torch.device):
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1, device=device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates, _ = D(interpolates)
    
    grad_outputs = torch.ones_like(d_interpolates, device=device)
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gp = ((gradient_norm - 1) ** 2).mean()
    return gp

class Trainer:
    def __init__(self, cfg: CFG, device: torch.device):
        self.cfg = cfg
        self.device = device

        self.G = Generator(nz=cfg.nz, ngf=cfg.ngf, nc=cfg.nc).to(device)
        self.D = Discriminator(nc=cfg.nc, ndf=cfg.ndf).to(device)
        self.E = Encoder(nz=cfg.nz, ngf=cfg.ngf, nc=cfg.nc).to(device)

        self.G.apply(weights_init_normal)
        self.D.apply(weights_init_normal)
        self.E.apply(weights_init_normal)

        self.opt_G = torch.optim.Adam(self.G.parameters(), lr=cfg.lr, betas=cfg.betas)
        self.opt_D = torch.optim.Adam(self.D.parameters(), lr=cfg.lr, betas=cfg.betas)
        self.opt_E = torch.optim.Adam(self.E.parameters(), lr=cfg.lr, betas=cfg.betas)

        self.mu_score = 0.0
        self.sigma_score = 1.0
        self.min_score = 0.0
        self.max_score = 1.0
        self.q95 = 0.0
        
        self.global_models = {}
        self.global_model_weights = {
            'spatial_transformer': 0.5,
        }

    def train_gan(self, dataloader: DataLoader, epochs: int = 50, save_every: int = 5):
        cfg = self.cfg
        iters = 0
        
        for epoch in range(epochs):
            pbar = tqdm(dataloader, desc=f"WGAN-GP Epoch {epoch+1}/{epochs}")
            
            for i, (real_patches, _) in enumerate(pbar):
                real_patches = real_patches.to(self.device)
                bsz = real_patches.size(0)

                self.opt_D.zero_grad()
                z = torch.randn(bsz, cfg.nz, 1, 1, device=self.device)
                fake = self.G(z)
                real_validity, _ = self.D(real_patches)
                fake_validity, _ = self.D(fake.detach())
                
                d_loss = fake_validity.mean() - real_validity.mean()
                gp = compute_gradient_penalty(self.D, real_patches.data, fake.data, self.device)
                d_loss = d_loss + cfg.lambda_gp * gp
                
                d_loss.backward()
                self.opt_D.step()

                if (i + 1) % cfg.n_critic == 0:
                    self.opt_G.zero_grad()
                    z = torch.randn(bsz, cfg.nz, 1, 1, device=self.device)
                    gen = self.G(z)
                    g_loss = -self.D(gen)[0].mean()
                    g_loss.backward()
                    self.opt_G.step()

                iters += 1
                if iters % 100 == 0:
                    pbar.set_postfix({
                        "d_loss": f"{d_loss.item():.4f}",
                        "gp": f"{gp.item():.4f}",
                        "g_loss": f"{g_loss.item() if 'g_loss' in locals() else 0:.4f}"
                    })

            if (epoch + 1) % save_every == 0 or epoch == epochs - 1:
                sample_path = os.path.join(cfg.out_dir, f"sample_epoch_{epoch+1}.png")
                self.save_samples(16, sample_path)
                
                ckpt = {
                    "epoch": epoch + 1,
                    "G": self.G.state_dict(),
                    "D": self.D.state_dict(),
                    "opt_G": self.opt_G.state_dict(),
                    "opt_D": self.opt_D.state_dict()
                }
                torch.save(ckpt, os.path.join(cfg.ckpt_dir, f"gan_epoch_{epoch+1}.pth"))
                torch.save(ckpt, os.path.join(cfg.ckpt_dir, f"gan_latest.pth"))

    def save_samples(self, n_samples: int, path: str):
        self.G.eval()
        with torch.no_grad():
            z = torch.randn(n_samples, self.cfg.nz, 1, 1, device=self.device)
            imgs = self.G(z)
            imgs = (imgs + 1.0) / 2.0
            vutils.save_image(imgs, path, nrow=4)
        self.G.train()

    def train_encoder(self, dataloader: DataLoader, epochs: int = 50):
        for epoch in range(epochs):
            total_loss = 0.0
            total_rec_loss = 0.0
            total_fm_loss = 0.0
            
            pbar = tqdm(dataloader, desc=f"Encoder Epoch {epoch+1}/{epochs}")
            
            for imgs, _ in pbar:
                imgs = imgs.to(self.device)
                self.opt_E.zero_grad()

                z = self.E(imgs)
                z_in = z.view(z.size(0), z.size(1), 1, 1)
                recon = self.G(z_in)

                loss_rec = F.mse_loss(recon, imgs)

                _, feats_real = self.D(imgs)
                _, feats_fake = self.D(recon)
                
                fm_loss = 0.0
                for fr, ff in zip(feats_real, feats_fake):
                    fr_flat = fr.view(fr.size(0), -1)
                    ff_flat = ff.view(ff.size(0), -1)
                    fm_loss = fm_loss + F.mse_loss(ff_flat, fr_flat.detach())

                loss = loss_rec + self.cfg.kappa * fm_loss

                loss.backward()
                self.opt_E.step()

                total_loss += loss.item()
                total_rec_loss += loss_rec.item()
                total_fm_loss += fm_loss.item()
                
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "rec": f"{loss_rec.item():.4f}",
                    "fm": f"{fm_loss.item():.4f}"
                })

            avg_loss = total_loss / len(dataloader)
            avg_rec = total_rec_loss / len(dataloader)
            avg_fm = total_fm_loss / len(dataloader)
            
            print(f"Epoch {epoch+1}: Loss={avg_loss:.4f}, Rec={avg_rec:.4f}, FM={avg_fm:.4f}")

            if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                ckpt = {
                    "epoch": epoch + 1,
                    "E": self.E.state_dict(),
                    "opt_E": self.opt_E.state_dict(),
                    "loss": avg_loss,
                    "rec_loss": avg_rec,
                    "fm_loss": avg_fm
                }
                torch.save(ckpt, os.path.join(self.cfg.ckpt_dir, f"encoder_epoch_{epoch+1}.pth"))
                torch.save(ckpt, os.path.join(self.cfg.ckpt_dir, f"encoder_latest.pth"))
                
                self.test_reconstruction(dataloader, epoch + 1)

    def test_reconstruction(self, dataloader: DataLoader, epoch: int):
        self.G.eval()
        self.E.eval()
        
        with torch.no_grad():
            imgs, _ = next(iter(dataloader))
            imgs = imgs[:8].to(self.device)
            
            z = self.E(imgs)
            z_in = z.view(z.size(0), z.size(1), 1, 1)
            recon = self.G(z_in)
            
            mse = F.mse_loss(recon, imgs).item()
            if mse > 0:
                psnr = 20 * torch.log10(torch.tensor(1.0) / torch.sqrt(torch.tensor(mse))).item()
            else:
                psnr = float('inf')
            
            comparison = torch.cat([imgs, recon])
            comparison = (comparison + 1.0) / 2.0
            
            save_path = os.path.join(self.cfg.out_dir, f"reconstruction_epoch_{epoch}.png")
            vutils.save_image(comparison, save_path, nrow=8)
            
            print(f"Epoch {epoch}: Reconstruction PSNR = {psnr:.2f} dB")
        
        self.G.train()
        self.E.train()

    def load_gan(self, ckpt_path: str):
        ck = torch.load(ckpt_path, map_location=self.device)
        self.G.load_state_dict(ck['G'])
        self.D.load_state_dict(ck['D'])
        if 'opt_G' in ck:
            self.opt_G.load_state_dict(ck['opt_G'])
        if 'opt_D' in ck:
            self.opt_D.load_state_dict(ck['opt_D'])

    def load_encoder(self, ckpt_path: str):
        ck = torch.load(ckpt_path, map_location=self.device)
        self.E.load_state_dict(ck['E'])
        if 'opt_E' in ck:
            self.opt_E.load_state_dict(ck['opt_E'])

    def compute_score_stats(self, dataloader: DataLoader):
        scores = []
        self.G.eval()
        self.E.eval()
        self.D.eval()
        
        with torch.no_grad():
            for imgs, _ in tqdm(dataloader, desc="Processing training images"):
                imgs = imgs.to(self.device)
                raw_scores, _, _ = self.anomaly_score(
                    imgs, use_global=False, global_model_name=None, normalize=False
                )
                scores.extend(raw_scores)
        
        scores = np.array(scores)
        
        self.mu_score = scores.mean()
        self.sigma_score = scores.std() if scores.std() > 0 else 1.0
        self.min_score = scores.min()
        self.max_score = scores.max()
        self.q95 = np.percentile(scores, 95)
        self.q99 = np.percentile(scores, 99)
        
        print("ANOMALY SCORE STATISTICS (Training Set)")
        print(f"Mean (μ):            {self.mu_score:.6f}")
        print(f"Std (σ):             {self.sigma_score:.6f}")
        print(f"Min:                 {self.min_score:.6f}")
        print(f"Max:                 {self.max_score:.6f}")
        print(f"95th percentile:     {self.q95:.6f}")
        print(f"99th percentile:     {self.q99:.6f}")
        print(f"Range:               [{self.min_score:.6f}, {self.max_score:.6f}]")
        
        self.plot_score_distribution(scores, "train_scores_distribution.png")
        
        self.G.train()
        self.E.train()
        self.D.train()
        
        return scores
    
    def global_features(self, imgs, model_name):
        global_model = self.build_global_model(model_name)
        global_model.eval()

        with torch.no_grad():
            feats = global_model((imgs + 1) / 2) 

        if isinstance(feats, tuple):          # If model returns tuple (e.g., Spatial Transformer)
            feats = feats[0]   # take the main embedding

        return feats

    def compute_global_feature_stats(self, dataloader, model_name='spatial_transformer'):
        feats_all = []
        if(model_name != 'none'):
            for imgs, _ in tqdm(dataloader, desc="Building global feature distribution"):
                imgs = imgs.to(self.device)

                feats = self.global_features(imgs, model_name)
                feats_all.append(feats.cpu())

            feats_all = torch.cat(feats_all, dim=0)   # shape: N × D
            self.global_mean = feats_all.mean(dim=0)
            self.global_cov = torch.cov(feats_all.T) + 1e-5 * torch.eye(feats_all.shape[1])

            self.global_cov_inv = torch.inverse(self.global_cov)         # inverse covariance

            print(" mean:", self.global_mean.shape)
            print(" cov:", self.global_cov.shape)

    def global_anomaly_score(self, img_tensor, model_name='spatial_transformer', norm_method='none'):
        feats = self.global_features(img_tensor, model_name)  # shape: 1 × D
        feats = feats.squeeze(0).cpu()

        # 2) Compute raw Mahalanobis distance score
        delta = feats - self.global_mean
        raw_score = torch.sqrt(delta @ self.global_cov_inv @ delta.T).item()

        # 3) Normalization
        if norm_method == 'none':
            return raw_score

        elif norm_method == 'minmax':             # Avoid division by zero
            denom = (self.max_score - self.min_score) if (self.max_score - self.min_score) != 0 else 1e-9
            norm_score = (raw_score - self.min_score) / denom
            return float(norm_score)
        else:
            return raw_score

    def plot_score_distribution(self, scores: np.ndarray, filename: str):
        plt.figure(figsize=(12, 6))
        plt.hist(scores, bins=50, alpha=0.7, color='blue', edgecolor='black', density=True)
        plt.axvline(self.mu_score, color='red', linestyle='--', linewidth=2, label=f'Mean (μ): {self.mu_score:.4f}')
        plt.axvline(self.q95, color='green', linestyle='--', linewidth=2, label=f'95th percentile: {self.q95:.4f}')
        plt.axvline(self.q99, color='orange', linestyle='--', linewidth=2, label=f'99th percentile: {self.q99:.4f}')
        
        kde = gaussian_kde(scores)
        x_range = np.linspace(scores.min(), scores.max(), 1000)
        plt.plot(x_range, kde(x_range), 'r-', linewidth=2, label='KDE')
        
        plt.xlabel('Anomaly Score', fontsize=12)
        plt.ylabel('Density', fontsize=12)
        plt.title('Distribution of Training Anomaly Scores', fontsize=14)
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        save_path = os.path.join(self.cfg.out_dir, filename)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
        
    def build_global_model(self, model_name: str = 'spatial_transformer'):
        if model_name in self.global_models:
            return self.global_models[model_name]
                
        if model_name == 'spatial_transformer':
            class SpatialRelationModel(nn.Module):
                def __init__(self, feature_dim=512):
                    super().__init__()
                    self.cnn = nn.Sequential(
                        nn.Conv2d(3, 64, 3, padding=1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        
                        nn.Conv2d(64, 128, 3, padding=1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                        
                        nn.Conv2d(128, 256, 3, padding=1),
                        nn.BatchNorm2d(256),
                        nn.ReLU(),
                        nn.MaxPool2d(2),
                    )
                    
                    self.spatial_attention = nn.Sequential(
                        nn.Conv2d(256, 128, 1),
                        nn.BatchNorm2d(128),
                        nn.ReLU(),
                        nn.Conv2d(128, 64, 1),
                        nn.BatchNorm2d(64),
                        nn.ReLU(),
                        nn.Conv2d(64, 1, 1),
                        nn.Sigmoid()
                    )
                    self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
                    
                def forward(self, x):
                    features = self.cnn(x)
                    attention_map = self.spatial_attention(features)
                    weighted_features = features * attention_map
                    global_features = self.global_pool(weighted_features)
                    global_features = global_features.view(global_features.size(0), -1)
                    return global_features, attention_map
            
            model = SpatialRelationModel().to(self.device)
        
        model = model.to(self.device)
        for param in model.parameters():
            param.requires_grad = False
        
        model.eval()
        self.global_models[model_name] = model
        return model
   
    def compute_global_anomaly_score(self, imgs: torch.Tensor, recon: torch.Tensor, global_model: nn.Module, model_name: str = 'spatial_transformer'):
        batch_size = imgs.size(0)
        
        with torch.no_grad():
            if model_name == 'spatial_transformer':
                gf_real, attn_real = global_model(imgs)
                gf_fake, attn_fake = global_model(recon)
                
                attn_diff = F.mse_loss(attn_fake, attn_real, reduction='none')
                attn_score = attn_diff.view(batch_size, -1).mean(dim=1)
                
                cos_sim = F.cosine_similarity(gf_real, gf_fake, dim=1)
                feature_score = 1.0 - cos_sim
                
                return feature_score + 0.1 * attn_score
            else:
                imgs_01 = (imgs + 1.0) / 2.0
                recon_01 = (recon + 1.0) / 2.0
                
                mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)
                
                imgs_norm = (imgs_01 - mean) / std
                recon_norm = (recon_01 - mean) / std
                
                gf_real = global_model(imgs_norm)
                gf_fake = global_model(recon_norm)
            
            if len(gf_real.shape) > 2:
                gf_real = gf_real.view(batch_size, -1)
                gf_fake = gf_fake.view(batch_size, -1)
            
            cos_sim = F.cosine_similarity(gf_real, gf_fake, dim=1)
            global_loss = 1.0 - cos_sim
            
            return global_loss

    def anomaly_score(self, imgs: torch.Tensor, use_global: bool = False, global_model_name: str = 'spatial_transformer', normalize: bool = True, method: str = 'percentile', **kwargs):
        self.G.eval()
        self.E.eval()
        self.D.eval()
        
        with torch.no_grad():
            z = self.E(imgs)
            z_in = z.view(z.size(0), z.size(1), 1, 1)
            recon = self.G(z_in)
            
            batch_size = imgs.size(0)
            rec_mse = F.mse_loss(recon, imgs, reduction='none')
            A_R = rec_mse.view(batch_size, -1).mean(dim=1)
            
            _, feats_real = self.D(imgs)
            _, feats_fake = self.D(recon)
            
            A_D = torch.zeros(batch_size, device=self.device)
            for fr, ff in zip(feats_real, feats_fake):
                fr_flat = fr.view(batch_size, -1)
                ff_flat = ff.view(batch_size, -1)
                layer_diff = F.mse_loss(ff_flat, fr_flat, reduction='none').mean(dim=1)
                A_D = A_D + layer_diff
            
            A_D = A_D / len(feats_real)
            
            raw_score = A_R + self.cfg.kappa * A_D
            
            if use_global and global_model_name and global_model_name != 'none':
                global_model = self.build_global_model(global_model_name)
                global_loss = self.compute_global_anomaly_score(imgs, recon, global_model, global_model_name)
                
                weight = self.global_model_weights.get(global_model_name, 0.5)
                raw_score = raw_score + weight * global_loss
            
            raw_score_np = raw_score.cpu().numpy()
            
            if normalize and hasattr(self, 'mu_score') and self.sigma_score > 0:                    
                if method == 'zscore':
                    z_scores = (raw_score_np - self.mu_score) / (self.sigma_score + 1e-8)
                    normalized = np.log1p(np.exp(z_scores))
                    
                elif method == 'minmax':
                    normalized = (raw_score_np - self.min_score) / (self.max_score - self.min_score + 1e-8)
                    normalized = np.clip(normalized, 0, 1)
                    
                elif method == 'raw':
                    normalized = raw_score_np
            else:
                normalized = raw_score_np
        
        return normalized, raw_score_np, recon.cpu()


def evaluate_single_model(trainer: Trainer, data_root: str, category: str, device: torch.device, global_model_name: str = 'none'):
    eval_dir = os.path.join(cfg.out_dir, "evaluation", category, global_model_name)
    os.makedirs(eval_dir, exist_ok=True)
    
    test_categories = ["good", "logical_anomalies"]
    
    all_results = []
    y_true = []
    y_scores = []
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    for test_cat in test_categories:
        folder_path = Path(data_root) / category / "test" / test_cat
        
        if not folder_path.exists():
            print(f"Warning: {folder_path} does not exist, skipping...")
            continue
        
        exts = ('*.png', '*.jpg', '*.jpeg')
        files = []
        for e in exts:
            files.extend(sorted(glob.glob(str(folder_path / e))))
                
        for file_path in tqdm(files, desc=f"{test_cat} ({global_model_name})"):
            try:
                img = Image.open(file_path).convert("RGB")
                img_resized = img.resize((cfg.image_resize, cfg.image_resize))
                img_tensor = transform(img_resized).unsqueeze(0).to(device)
                
                _, _, H, W = img_tensor.shape

                # 1) PATCH-BASED LOCAL FANOGAN ANOMALY SCORE
                patch_scores = []
                patch_locations = []
                
                for top in range(0, H - cfg.patch_size + 1, cfg.patch_stride):
                    for left in range(0, W - cfg.patch_size + 1, cfg.patch_stride):
                        patch = img_tensor[..., top:top+cfg.patch_size, left:left+cfg.patch_size]
                        
                        normalized_score, raw_score, _ = trainer.anomaly_score(
                            patch, use_global=True, normalize=True, method='raw'
                        )
                        
                        patch_scores.append(float(normalized_score[0]))
                        patch_locations.append((top, left))
                
                if patch_scores:
                    patch_score_image = np.mean(patch_scores)
                else:
                    patch_score_image = 0.0

                # 2) GLOBAL SCORE (FULL-IMAGE)
                if global_model_name != "none":
                        global_score = trainer.global_anomaly_score(img_tensor, global_model_name,'minmax')
                else:
                    global_score = 0.0
                
                # 3) FINAL ANOMALY SCORE (COMBINED)
                w_patch = 0.6
                w_global = 1.0
                
                image_score = w_patch * patch_score_image + w_global * global_score
                if isinstance(patch_score_image, torch.Tensor):
                    patch_score_image = float(patch_score_image.cpu())

                if isinstance(global_score, torch.Tensor):
                    global_score = float(global_score.cpu())

                image_score = float(image_score)
                result = {
                    'image_path': file_path,
                    'category': test_cat,
                    'image_score': image_score,
                    'patch_scores': patch_scores,
                    'patch_locations': patch_locations,
                    'label': 0 if test_cat == 'good' else 1
                }
                
                all_results.append(result)
                y_true.append(int(result['label']))
                y_scores.append(image_score)
                
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                continue
    
    y_true = np.array(y_true)
    y_scores = np.array(y_scores)    

    if len(np.unique(y_true)) > 1:
        auc_score, ap_score, optimal_threshold = plot_evaluation_curves(y_true, y_scores, eval_dir, category, global_model_name)
        
        y_pred = (y_scores > optimal_threshold).astype(int)
        
        tp = np.sum((y_true == 1) & (y_pred == 1))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fn = np.sum((y_true == 1) & (y_pred == 0))
        
        accuracy = (tp + tn) / len(y_true) if len(y_true) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\nEVALUATION RESULTS - {category} ({global_model_name})")
        print(f"AUC:                      {auc_score:.4f}")
        print(f"Average Precision:        {ap_score:.4f}")
        print(f"Optimal Threshold:        {optimal_threshold:.4f}")
        print(f"\nClassification Metrics:")
        print(f"Accuracy:                 {accuracy:.4f}")
        print(f"Precision:                {precision:.4f}")
        print(f"Recall (Sensitivity):     {recall:.4f}")
        print(f"F1 Score:                 {f1:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"True Positives:           {tp}")
        print(f"False Positives:          {fp}")
        print(f"True Negatives:           {tn}")
        print(f"False Negatives:           {fn}")
        
        scores_good = [r['image_score'] for r in all_results if r['category'] == 'good']
        scores_anomaly = [r['image_score'] for r in all_results if r['category'] == 'logical_anomalies']
        
        plot_score_distributions(
            scores_good, scores_anomaly, optimal_threshold, os.path.join(eval_dir, f'score_distributions_{global_model_name}.png')
        )
        
        results_df = pd.DataFrame([{
            'image_path': r['image_path'],
            'category': r['category'],
            'image_score': r['image_score'],
            'label': r['label'],
            'prediction': 1 if r['image_score'] > optimal_threshold else 0,
            'correct': (1 if r['image_score'] > optimal_threshold else 0) == r['label']
        } for r in all_results])
        
        results_df.to_csv(os.path.join(eval_dir, f'results_{global_model_name}.csv'), index=False)
                
        top_anomalies = sorted([r for r in all_results if r['category'] == 'logical_anomalies'], key=lambda x: x['image_score'], reverse=True)[:5]
        
        for i, result in enumerate(top_anomalies):
            try:
                save_path = os.path.join(eval_dir, f'top_anomaly_{i+1}_{Path(result["image_path"]).stem}.png')
                plot_patch_heatmap(
                    result['image_path'], result['patch_scores'], result['patch_locations'],
                    result['image_score'], optimal_threshold, save_path, cfg, global_model_name
                )
            except Exception as e:
                print(f"Error generating heatmap for {result['image_path']}: {e}")
        
        normal_images = sorted([r for r in all_results if r['category'] == 'good'], key=lambda x: x['image_score'])[:3]
        
        for i, result in enumerate(normal_images):
            try:
                save_path = os.path.join(eval_dir, f'normal_{i+1}_{Path(result["image_path"]).stem}.png')
                plot_patch_heatmap(
                    result['image_path'], result['patch_scores'], result['patch_locations'],
                    result['image_score'], optimal_threshold, save_path, cfg, global_model_name
                )
            except Exception as e:
                print(f"Error generating heatmap for {result['image_path']}: {e}")
                
        return auc_score, ap_score, optimal_threshold, all_results
    else:
        return 0.0, 0.0, 0.0, all_results

def parse_args():
    parser = argparse.ArgumentParser(description="Enhanced f-AnoGAN for Logical Anomalies")
    parser.add_argument("--data_root", type=str, required=True, help="Path to MVTec LOCO dataset")
    parser.add_argument("--category", type=str, required=True, help="Category name (e.g., breakfast_box)")
    parser.add_argument("--phase", type=str, choices=["train_gan", "train_encoder", "eval"], default="train_gan", help="Training phase")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=None, help="Batch size")
    parser.add_argument("--ckpt", type=str, default=None, help="GAN checkpoint path")
    parser.add_argument("--enc_ckpt", type=str, default=None, help="Encoder checkpoint path")
    parser.add_argument("--global_model", type=str, choices=["none", "spatial_transformer"], default="none", help="Global model")
    parser.add_argument("--max_patches", type=int, default=cfg.max_patches, help="Maximum number of patches to precompute")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.batch:
        cfg.batch_size = args.batch
    if args.epochs:
        if args.phase == "train_gan":
            cfg.epochs_gan = args.epochs
        elif args.phase == "train_encoder":
            cfg.epochs_enc = args.epochs
    if args.max_patches:
        cfg.max_patches = args.max_patches
    
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    trainer = Trainer(cfg, device)
         
    if args.phase == "train_gan":        
        train_ds = MVTecLOCO_PatchDataset(
            args.data_root, args.category, mode='train',
            patch_size=cfg.patch_size, stride=cfg.patch_stride,
            image_resize=cfg.image_resize, transform=base_transform,
            precompute=cfg.patches_precompute, max_patches=cfg.max_patches
        )
        
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True,
            num_workers=cfg.num_workers, pin_memory=True
        )
        
        trainer.train_gan(train_loader, epochs=cfg.epochs_gan)        
    elif args.phase == "train_encoder":
        if args.ckpt is None:
            raise RuntimeError("You must provide --ckpt for GAN before training encoder")
                
        train_ds = MVTecLOCO_PatchDataset(
            args.data_root, args.category, mode='train',
            patch_size=cfg.patch_size, stride=cfg.patch_stride,
            image_resize=cfg.image_resize, transform=base_transform,
            precompute=cfg.patches_precompute, max_patches=cfg.max_patches
        )
        
        train_loader = DataLoader(
            train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True
        )
        
        trainer.load_gan(args.ckpt)        
        trainer.train_encoder(train_loader, epochs=cfg.epochs_enc)
        
    elif args.phase == "eval":
        if args.ckpt is None or args.enc_ckpt is None:
            raise RuntimeError("You must provide --ckpt/enc_ckpt for GAN evaluation")
 
        trainer.load_gan(args.ckpt)
        trainer.load_encoder(args.enc_ckpt)
        
        if not hasattr(trainer, 'mu_score') or abs(trainer.mu_score) < 1e-9:            
            train_ds = MVTecLOCO_PatchDataset(
                args.data_root, args.category, mode='train',
                patch_size=cfg.patch_size, stride=cfg.patch_stride,
                image_resize=cfg.image_resize, transform=base_transform,
                precompute=cfg.patches_precompute, max_patches=cfg.max_patches
            )
            
            train_loader = DataLoader(
                train_ds, batch_size=cfg.batch_size, shuffle=False,
                num_workers=cfg.num_workers
            )

            trainer.compute_score_stats(train_loader)
        
            train_dataset_full =  MVTecLOCO_PatchDataset(
                args.data_root, args.category, mode='train1', image_resize=cfg.image_resize
            )

            train_loader_full = DataLoader(
                train_dataset_full, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers
            )
    
        trainer.compute_global_feature_stats(train_loader_full, model_name=args.global_model)
        auc_score, ap_score, optimal_threshold, all_results = evaluate_single_model(
            trainer, args.data_root, args.category, device, args.global_model
        )
    else:
        raise ValueError(f"Unknown phase: {args.phase}")

if __name__ == "__main__":
    main()