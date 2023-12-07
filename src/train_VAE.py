# IMPORTS
import torch, os, gc, cv2, time, random, warnings, matplotlib

import torch.nn.functional as Fun
import matplotlib.pyplot as plt
import numpy as np

from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from CBAM_VAE import AttentionVAE
import utils

matplotlib.use('tkagg')
warnings.filterwarnings("ignore")
torch.cuda.empty_cache()
gc.collect()
unloader = transforms.ToPILImage()

def compute_epoch_loss(model, data_loader, loss_fn, device):
    """Compute the loss for a given model and data loader."""
    model.eval()
    total_loss, num_examples = 0., 0
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = loss_fn(outputs, inputs, reduction='sum')
            num_examples += inputs.size(0)
            total_loss += loss
    return total_loss / num_examples

class ImageDataset(Dataset):
    """Custom Dataset class for loading images from a directory."""

    def __init__(self, img_dir='./dataset5/'):
        self.img_path = [os.path.join(img_dir, file) for file in os.listdir(img_dir)]
        self.transform = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        img_file = self.img_path[idx]
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_tensor = self.transform(img)
        return (img_tensor, img_tensor)

def train(model, optimizer, device, train_loader, num_epochs, loss_fn, logging_interval, reconstruction_term_weight, save_model):
    """Train the VAE model."""
    log_dict = {
        'train_combined_loss_per_batch': [],
        'train_combined_loss_per_epoch': [],
        'train_reconstruction_loss_per_batch': [],
        'train_kl_loss_per_batch': []
    }

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (X_img, Y) in enumerate(train_loader):
            X_img, Y = X_img.to(device), Y.to(device)
            optimizer.zero_grad()
            z_a, z_mean, z_log_var, X_decoded = model(X_img)

            kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean ** 2 - torch.exp(z_log_var), axis=1)
            kl_div_mean = kl_div.mean()

            pixelwise = loss_fn(X_decoded, Y, reduction='none').view(X_img.size(0), -1).sum(axis=1).mean()
            loss = reconstruction_term_weight * pixelwise + kl_div_mean

            loss.backward()
            optimizer.step()

            log_dict['train_combined_loss_per_batch'].append(loss.item())
            log_dict['train_reconstruction_loss_per_batch'].append(pixelwise.item())
            log_dict['train_kl_loss_per_batch'].append(kl_div_mean.item())

            if batch_idx % logging_interval == 0:
                print(f'Epoch: {epoch + 1}/{num_epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss:.4f} (Pixelwise: {pixelwise:.4f} | KL Div: {kl_div_mean:.4f})')

        # Save model periodically
        if epoch % 100 == 0:
            torch.save(model.state_dict(), f'temp_vae_models/temp_model_e{epoch}_l{loss:.2f}.pt')

        # Compute loss per epoch
        train_loss = compute_epoch_loss(model, train_loader, loss_fn, device)
        log_dict['train_combined_loss_per_epoch'].append(train_loss)
        print(f'*** Epoch: {epoch + 1}/{num_epochs} | Loss: {train_loss:.3f}')

    print(f'Total Training Time: {(time.time() - start_time) / 60:.2f} min')
    if save_model is not None:
        torch.save(model.state_dict(), save_model)

    return log_dict