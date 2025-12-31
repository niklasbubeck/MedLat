#!/usr/bin/env python3
"""
Training script for diffusion generator models.
Converted from example_generator.ipynb
"""

from medtok import available_models, get_model, MODEL_REGISTRY
from medtok.diffusion import create_gaussian_diffusion
from medtok.modules.wrapper import GenWrapper

import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for script execution
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST, CIFAR100, CIFAR10
from medmnist import PathMNIST, BreastMNIST
from tqdm import tqdm
import torch.nn.functional as F
# from dataset import MergedMedMNIST  # Uncomment if needed
from medmnist import INFO


def train(model,
          train_set,
          test_set,
          batch_size,
          num_epochs,
          evaluation_iterations,
          save_samples=True):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    trainloader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=8)
    testloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=8)

    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    diffusion = create_gaussian_diffusion(
        steps=1000,
        learn_sigma=True,
        sigma_small=False,
        noise_schedule="linear",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing=""
    )

    diffusion_eval = create_gaussian_diffusion(
        steps=1000,
        learn_sigma=True,
        sigma_small=False,
        noise_schedule="linear",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=False,
        rescale_learned_sigmas=False,
        timestep_respacing="250"
    )

    train_losses = []
    evaluation_losses = []

    steps_per_epoch = len(trainloader)
    total_steps = steps_per_epoch * num_epochs
    step_counter = 0
    train_loss_running = []
    evaluation_loss_running = []

    with tqdm(total=num_epochs, desc="Epochs") as epoch_bar:
        for epoch in range(num_epochs):
            model.train()
            with tqdm(total=steps_per_epoch, desc=f"Epoch {epoch+1}/{num_epochs} | loss: N/A", leave=False) as pbar:
                for images, labels in trainloader:
                    
                    images, labels = images.to(device), labels.to(device)
                    cond = dict(y=labels.squeeze(1))  # Uncomment for class conditioning

                    t = torch.randint(0, diffusion.num_timesteps, (images.shape[0],), device=images.device)
                    z = model.vae_encode(images)
                    
                    terms = diffusion.training_losses(model, z, t, model_kwargs=cond)
                    loss = terms["loss"].mean()

                    # Update tqdm description with current loss value
                    pbar.set_description(f"Epoch {epoch+1}/{num_epochs} | loss: {loss.item():.4f}")

                    train_loss_running.append(loss.item())

                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()

                    step_counter += 1
                    pbar.update(1)

                    # Evaluate every evaluation_iterations global step
                    if step_counter % evaluation_iterations == 0:
                        model.eval()
                        evaluation_loss_running.clear()
                        example_shown = False
                        with torch.no_grad():
                            # Use explicit shape for sampling instead of z.shape
                            sample_shape = z.shape
                            sample = diffusion_eval.p_sample_loop(
                                model, 
                                sample_shape, 
                                clip_denoised=False, 
                                progress=True, 
                                model_kwargs=cond, 
                                temperature=1.0
                            )
                            out = model.vae_decode(sample)
                        
                        # Visualize generated sample
                        if save_samples:
                            to_show = out[0].detach().cpu().numpy()
                            if to_show.ndim == 3 and to_show.shape[0] in [1, 3]:
                                # (C,H,W) format, move channel last if necessary
                                to_show = np.transpose(to_show, (1, 2, 0))
                            elif to_show.ndim == 2:
                                pass  # Already H x W

                            # Clamp to [0,1] for display
                            plt.figure(figsize=(6, 6))
                            plt.imshow(np.clip(to_show, 0, 1), cmap=None if to_show.ndim == 3 else "gray")
                            plt.axis('off')
                            plt.title(f"Step {step_counter} | Epoch {epoch+1}")
                            plt.savefig(f"sample_step_{step_counter}.png", bbox_inches='tight', dpi=100)
                            plt.close()

                        train_loss_val = np.mean(train_loss_running) if train_loss_running else 0.0
                        evaluation_loss_val = np.mean(evaluation_loss_running) if evaluation_loss_running else 0.0
                        train_losses.append(train_loss_val)
                        evaluation_losses.append(evaluation_loss_val)
                        train_loss_running = []
                        model.train()
            
            epoch_bar.update(1)
            

    print("Final Training Loss", train_losses[-1] if train_losses else "No log")
    print("Final Evaluation Loss", evaluation_losses[-1] if evaluation_losses else "No log")

    return model, train_losses, evaluation_losses


def main():
    # Set random seeds for reproducibility
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    # Prep Dataset
    tensor_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    # Dataset options - uncomment the one you want to use
    # train_set = MergedMedMNIST(root="/vol/miltank/users/bubeckn/MedMNIST", split="train", transform=tensor_transforms, download=False, size=224)
    # test_set = MergedMedMNIST(root="/vol/miltank/users/bubeckn/MedMNIST", split="test", transform=tensor_transforms, download=False, size=224)

    train_set = PathMNIST(root="/vol/miltank/users/bubeckn/MedMNIST", split="train", transform=tensor_transforms, download=False, as_rgb=True, size=224)
    test_set = PathMNIST(root="/vol/miltank/users/bubeckn/MedMNIST", split="test", transform=tensor_transforms, download=False, as_rgb=True, size=224)

    # train_set = CIFAR10(root=".", train=True, transform=tensor_transforms, download=True)
    # test_set = CIFAR10(root=".", train=False, transform=tensor_transforms, download=True)

    # Set Device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Print available models
    all_models = available_models()
    print(f"Available models: {len(all_models)} total")

    # Load tokenizer
    tokenizer = get_model(
        "discrete.simple_qinco.f4_d3_e8192", 
        img_size=224, 
        ckpt_path="ckpt.1.pt"
    )
    print("Tokenizer loaded")

    # Load generator
    generator = get_model(
        "dit.b_2", 
        img_size=224, 
        in_channels=tokenizer.embed_dim, 
        vae_stride=tokenizer.vae_stride,
    )
    print("Generator loaded")

    # Create wrapper
    wrapper = GenWrapper(generator, tokenizer)
    print("Wrapper created")

    # Train
    print("Starting training...")
    trained_model, train_losses, evaluation_losses = train(
        wrapper,
        train_set=train_set,
        test_set=test_set,
        batch_size=32,
        num_epochs=1000,
        evaluation_iterations=500,
        save_samples=True
    )

    print("Training completed!")
    print(f"Final training loss: {train_losses[-1] if train_losses else 'N/A'}")
    print(f"Final evaluation loss: {evaluation_losses[-1] if evaluation_losses else 'N/A'}")


if __name__ == "__main__":
    main()

