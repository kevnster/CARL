# IMPORTS
import torch
import torch.nn as nn

class TensorReshaper(nn.Module):
    """
    This module reshapes an input tensor to a new specified shape.
    The new shape is provided as arguments during the initialization.
    """
    def __init__(self, *shape):
        super(TensorReshaper, self).__init__()
        self.new_shape = shape

    def forward(self, input_tensor):
        # Reshape the input tensor to the new shape
        return input_tensor.view(self.new_shape)

class TensorTrimmer(nn.Module):
    """
    This module trims the edges of a 4D tensor.
    Specifically, it removes the last row and column from the last two dimensions of the tensor.
    """
    def __init__(self):
        super(TensorTrimmer, self).__init__()

    def forward(self, input_tensor):
        # Trim the last row and column from the tensor
        trimmed_tensor = input_tensor[:, :, :-1, :-1]
        return trimmed_tensor

class FocusChannel(nn.Module):
    """
    A module for applying channel-wise attention to a tensor, enhancing important features.
    Based on the concept of Convolutional Block Attention Module.
    """
    def __init__(self, num_channels, compression=16):
        super(FocusChannel, self).__init__()
        self.avg_pooling = nn.AdaptiveAvgPool2d(1)
        self.max_pooling = nn.AdaptiveMaxPool2d(1)

        self.compress = nn.Conv2d(num_channels, num_channels // compression, 1, bias=False)
        self.activate = nn.ReLU()
        self.expand = nn.Conv2d(num_channels // compression, num_channels, 1, bias=False)

        self.activate_sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        avg_pooled = self.expand(self.activate(self.compress(self.avg_pooling(input_tensor))))
        max_pooled = self.expand(self.activate(self.compress(self.max_pooling(input_tensor))))
        combined = avg_pooled + max_pooled
        return self.activate_sigmoid(combined)

class FocusSpatial(nn.Module):
    """
    A module for applying spatial attention to a tensor, emphasizing relevant spatial regions.
    Inspired by the Spatial Attention Module concept.
    """
    def __init__(self, filter_size=7):
        super(FocusSpatial, self).__init__()

        assert filter_size in (3, 7), 'Filter size must be either 3 or 7'
        pad = 3 if filter_size == 7 else 1

        self.spatial_filter = nn.Conv2d(2, 1, filter_size, padding=pad, bias=False)
        self.activate_sigmoid = nn.Sigmoid()

    def forward(self, input_tensor):
        avg_channel = torch.mean(input_tensor, dim=1, keepdim=True)
        max_channel, _ = torch.max(input_tensor, dim=1, keepdim=True)
        combined_channel = torch.cat([avg_channel, max_channel], dim=1)
        filtered = self.spatial_filter(combined_channel)
        return self.activate_sigmoid(filtered)

class EnhancedAttentionModule(nn.Module):
    """
    Enhanced Attention Module that combines channel and spatial attention mechanisms.
    This module helps in focusing on the most informative features both channel-wise and spatially.
    """
    def __init__(self, num_channels):
        super(EnhancedAttentionModule, self).__init__()

        self.channel_focus = FocusChannel(num_channels)
        self.spatial_focus = FocusSpatial()

    def forward(self, input_tensor):
        # Apply channel attention and then spatial attention
        channel_attended = self.channel_focus(input_tensor) * input_tensor
        spatially_attended = self.spatial_focus(channel_attended) * channel_attended

        return spatially_attended

class ChannelFocusModule(nn.Module):
    """
    Channel Focus Module that applies channel-wise attention to enhance feature representation.
    It helps the network to focus on the most important features across the channels.
    """
    def __init__(self, num_channels):
        super(ChannelFocusModule, self).__init__()

        self.channel_focus = FocusChannel(num_channels)

    def forward(self, input_tensor):
        # Apply channel attention
        channel_attended = self.channel_focus(input_tensor) * input_tensor

        return channel_attended

class SpatialFocusModule(nn.Module):
    """
    Spatial Focus Module that applies spatial attention.
    This module enhances the network's ability to focus on relevant spatial regions.
    """
    def __init__(self, num_channels):
        super(SpatialFocusModule, self).__init__()

        self.spatial_focus = FocusSpatial()

    def forward(self, input_tensor):
        # Apply spatial attention
        spatially_attended = self.spatial_focus(input_tensor) * input_tensor

        return spatially_attended

class AttentionVAE(nn.Module):
    """
    An enhanced Variational Autoencoder incorporating both Channel and Spatial Attention mechanisms.
    This architecture is designed for more efficient and specific image reconstruction tasks.
    Reference: Attention-Based Autoencoder for Image Reconstruction
    """
    def __init__(self, latent_dim):
        super(AttentionVAE, self).__init__()

        self.latent_dim = latent_dim

        self.encode = nn.Sequential(
            nn.Conv2d(3, 32, stride=2, kernel_size=3, bias=False, padding=1),
            SpatialFocusModule(32),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout2d(0.25),
            nn.Conv2d(32, 64, stride=2, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 64, stride=2, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 64, stride=2, kernel_size=3, bias=False, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout2d(0.25),
            nn.Conv2d(64, 64, stride=2, kernel_size=3, bias=False, padding=1),
            ChannelFocusModule(64),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Flatten(),
            nn.Linear(4096, 512),
            nn.BatchNorm1d(512),
            nn.PReLU()
        )

        self.latent_mean = nn.Sequential(
            nn.Linear(512, self.latent_dim)
        )
        self.latent_log_var = nn.Sequential(
            nn.Linear(512, self.latent_dim)
        )
        self.latent_to_feature = nn.Sequential(
            nn.Linear(self.latent_dim, 512),
            nn.BatchNorm1d(512),
            nn.PReLU()
        )

        self.decode = nn.Sequential(
            nn.Linear(512, 4096),
            nn.BatchNorm1d(4096),
            nn.PReLU(),
            TensorReshaper(-1, 64, 8, 8),
            nn.ConvTranspose2d(64, 64, stride=2, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout2d(0.25),
            nn.ConvTranspose2d(64, 64, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout2d(0.25),
            nn.ConvTranspose2d(64, 64, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.PReLU(),
            nn.Dropout2d(0.25),
            nn.ConvTranspose2d(64, 32, stride=2, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.PReLU(),
            nn.Dropout2d(0.25),
            nn.ConvTranspose2d(32, 3, stride=2, kernel_size=3, padding=1),
            TensorTrimmer(),
            nn.ReLU(inplace=True)
        )

    def encode_fn(self, observation):
        encoded = self.encode(observation)
        mean, log_var = self.latent_mean(encoded), self.latent_log_var(encoded)
        z = self.add_noise(mean, log_var)
        return z

    def navigation(self, current_obs, target_obs):
        current_z = self.encode_fn(current_obs)
        target_z = self.encode_fn(target_obs)
        return current_z, target_z

    def add_noise(self, mu, log_var):
        epsilon = torch.randn(mu.size(0), mu.size(1)).to(mu.device)
        z = mu + epsilon * torch.exp(log_var / 2.0)
        return z

    def forward(self, input_tensor):
        encoded = self.encode(input_tensor)
        mean, log_var = self.latent_mean(encoded), self.latent_log_var(encoded)
        z = self.add_noise(mean, log_var)
        feature_map = self.latent_to_feature(z)
        decoded = self.decode(feature_map)
        return z, mean, log_var, decoded

