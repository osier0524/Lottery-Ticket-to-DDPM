from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import os;

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
)

diffusion = GaussianDiffusion(
    model,
    image_size= 128,
    timesteps= 1000,
    sampling_timesteps= 250,
    loss_type= '11'
)

trainer = Trainer(
    diffusion,
    './dataset/train',
    train_batch_size= 2,
    train_lr= 8e-5,
    train_num_steps= 1000,
    gradient_accumulate_every = 2,
    ema_decay = 0.995,
    amp = True,
    calculate_fid = True
)

if __name__ == '__main__':
    os.environ['TORCH_HOME'] = 'D:\PyTorch'
    trainer.train()