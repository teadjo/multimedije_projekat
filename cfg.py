
class CFG:
    image_resize = 512
    patch_size = 128         
    patch_stride = 64         
    max_patches = 20000      
    patches_precompute = True 

    nz = 128                 
    nc = 3                    
    ngf = 64                  
    ndf = 64                  

    batch_size = 32
    num_workers = 4
    lr = 2e-4
    betas = (0.5, 0.9)       
    lambda_gp = 10.0         
    n_critic = 5             
    epochs_gan = 450
    epochs_enc = 150          
    kappa = 0.9            

    out_dir = "outputs"
    ckpt_dir = "checkpoints"

cfg = CFG()