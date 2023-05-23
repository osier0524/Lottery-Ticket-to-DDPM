## Lottery-Ticket-Hypothesis-in-DDPM

test.ipynb is an example that adapts Lottery Ticket Hypothesis to DDPM

## Dataset
I use [cifar10](https://www.kaggle.com/competitions/cifar-10/data), [cifar100](https://www.kaggle.com/datasets/fedesoriano/cifar100) as the dataset.
Images should be downloaded to a folder named dataset/XXX. Only need to download train set.

## Usage

Test case: Please use this block to test whether the program runs smoothly. This block can help save time.
```python
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
)

diffusion = GaussianDiffusion(
    model,
    image_size = 32,
    timesteps = 4,           # number of steps
    sampling_timesteps = 2,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type = 'l1'            # L1 or L2
)

pruning_trainer = Trainer(
    diffusion,
    train_batch_size = 20,
    train_lr = 8e-5,
    train_num_steps = 4,         # total training steps
    prune_end_iter = 5,            # pruning steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    save_and_sample_every=2
    ema_decay = 0.995,                # exponential moving average decay
    results_folder = "./results",
    amp = True,                       # turn on mixed precision
    calculate_fid = True,              # whether to calculate fid during training
    dataset = 'cifar10'
    arch_type = 'DDPM'
)

# Start to train
pruning_trainer.train()

```

1. Dataset: cifar10  Model: DDPM
```python
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

# Dataset: cifar10 Model: DDPM
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
)

diffusion = GaussianDiffusion(
    model,
    image_size = 32,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type = 'l1'            # L1 or L2
)

pruning_trainer = Trainer(
    diffusion,
    train_batch_size = 20,
    train_lr = 8e-5,
    train_num_steps = 2000,         # total training steps
    prune_end_iter = 35,            # pruning steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    results_folder = "./results",
    amp = True,                       # turn on mixed precision
    calculate_fid = True,              # whether to calculate fid during training
    dataset = 'cifar10'
    arch_type = 'DDPM'
)

# Start to train
pruning_trainer.train()
```

2. Dataset: cifar100 Model: DDPM
```python
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

# Dataset: cifar10 Model: DDPM
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8)
)

diffusion = GaussianDiffusion(
    model,
    image_size = 32,
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type = 'l1'            # L1 or L2
)

pruning_trainer = Trainer(
    diffusion,
    train_batch_size = 20,
    train_lr = 8e-5,
    train_num_steps = 2000,         # total training steps
    prune_end_iter = 35,            # pruning steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    results_folder = "./results",
    amp = True,                       # turn on mixed precision
    calculate_fid = True,              # whether to calculate fid during training
    dataset = 'cifar100',
    arch_type = 'DDPM'
)

# Start to train
pruning_trainer.train()
```

Samples and model checkpoints will be logged to `./results` periodically
Plots will be saved to `./plots`




## Citations

```bibtex
@inproceedings{NEURIPS2020_4c5bcfec,
    author      = {Ho, Jonathan and Jain, Ajay and Abbeel, Pieter},
    booktitle   = {Advances in Neural Information Processing Systems},
    editor      = {H. Larochelle and M. Ranzato and R. Hadsell and M.F. Balcan and H. Lin},
    pages       = {6840--6851},
    publisher   = {Curran Associates, Inc.},
    title       = {Denoising Diffusion Probabilistic Models},
    url         = {https://proceedings.neurips.cc/paper/2020/file/4c5bcfec8584af0d967f1ab10179ca4b-Paper.pdf},
    volume      = {33},
    year        = {2020}
}
```

```bibtex
@InProceedings{pmlr-v139-nichol21a,
    title       = {Improved Denoising Diffusion Probabilistic Models},
    author      = {Nichol, Alexander Quinn and Dhariwal, Prafulla},
    booktitle   = {Proceedings of the 38th International Conference on Machine Learning},
    pages       = {8162--8171},
    year        = {2021},
    editor      = {Meila, Marina and Zhang, Tong},
    volume      = {139},
    series      = {Proceedings of Machine Learning Research},
    month       = {18--24 Jul},
    publisher   = {PMLR},
    pdf         = {http://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf},
    url         = {https://proceedings.mlr.press/v139/nichol21a.html},
}
```

```bibtex
@inproceedings{kingma2021on,
    title       = {On Density Estimation with Diffusion Models},
    author      = {Diederik P Kingma and Tim Salimans and Ben Poole and Jonathan Ho},
    booktitle   = {Advances in Neural Information Processing Systems},
    editor      = {A. Beygelzimer and Y. Dauphin and P. Liang and J. Wortman Vaughan},
    year        = {2021},
    url         = {https://openreview.net/forum?id=2LdBqxc1Yv}
}
```

```bibtex
@article{Choi2022PerceptionPT,
    title   = {Perception Prioritized Training of Diffusion Models},
    author  = {Jooyoung Choi and Jungbeom Lee and Chaehun Shin and Sungwon Kim and Hyunwoo J. Kim and Sung-Hoon Yoon},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2204.00227}
}
```

```bibtex
@article{Karras2022ElucidatingTD,
    title   = {Elucidating the Design Space of Diffusion-Based Generative Models},
    author  = {Tero Karras and Miika Aittala and Timo Aila and Samuli Laine},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2206.00364}
}
```

```bibtex
@article{Song2021DenoisingDI,
    title   = {Denoising Diffusion Implicit Models},
    author  = {Jiaming Song and Chenlin Meng and Stefano Ermon},
    journal = {ArXiv},
    year    = {2021},
    volume  = {abs/2010.02502}
}
```

```bibtex
@misc{chen2022analog,
    title   = {Analog Bits: Generating Discrete Data using Diffusion Models with Self-Conditioning},
    author  = {Ting Chen and Ruixiang Zhang and Geoffrey Hinton},
    year    = {2022},
    eprint  = {2208.04202},
    archivePrefix = {arXiv},
    primaryClass = {cs.CV}
}
```

```bibtex
@article{Qiao2019WeightS,
    title   = {Weight Standardization},
    author  = {Siyuan Qiao and Huiyu Wang and Chenxi Liu and Wei Shen and Alan Loddon Yuille},
    journal = {ArXiv},
    year    = {2019},
    volume  = {abs/1903.10520}
}
```

```bibtex
@article{Salimans2022ProgressiveDF,
    title   = {Progressive Distillation for Fast Sampling of Diffusion Models},
    author  = {Tim Salimans and Jonathan Ho},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2202.00512}
}
```

```bibtex
@article{Ho2022ClassifierFreeDG,
    title   = {Classifier-Free Diffusion Guidance},
    author  = {Jonathan Ho},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2207.12598}
}
```

```bibtex
@article{Sunkara2022NoMS,
    title   = {No More Strided Convolutions or Pooling: A New CNN Building Block for Low-Resolution Images and Small Objects},
    author  = {Raja Sunkara and Tie Luo},
    journal = {ArXiv},
    year    = {2022},
    volume  = {abs/2208.03641}
}
```

```bibtex
@inproceedings{Jabri2022ScalableAC,
    title   = {Scalable Adaptive Computation for Iterative Generation},
    author  = {A. Jabri and David J. Fleet and Ting Chen},
    year    = {2022}
}
```

```bibtex
@article{Cheng2022DPMSolverPlusPlus,
    title   = {DPM-Solver++: Fast Solver for Guided Sampling of Diffusion Probabilistic Models},
    author  = {Cheng Lu and Yuhao Zhou and Fan Bao and Jianfei Chen and Chongxuan Li and Jun Zhu},
    journal = {NeuRips 2022 Oral},
    year    = {2022},
    volume  = {abs/2211.01095}
}
```

```bibtex
@inproceedings{Hoogeboom2023simpleDE,
    title   = {simple diffusion: End-to-end diffusion for high resolution images},
    author  = {Emiel Hoogeboom and Jonathan Heek and Tim Salimans},
    year    = {2023}
}
```

```bibtex
@misc{https://doi.org/10.48550/arxiv.2302.01327,
    doi     = {10.48550/ARXIV.2302.01327},
    url     = {https://arxiv.org/abs/2302.01327},
    author  = {Kumar, Manoj and Dehghani, Mostafa and Houlsby, Neil},
    title   = {Dual PatchNorm},
    publisher = {arXiv},
    year    = {2023},
    copyright = {Creative Commons Attribution 4.0 International}
}
```
