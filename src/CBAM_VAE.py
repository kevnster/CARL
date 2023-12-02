# IMPORTS
import torch
import torch.nn as nn
import torch.nn.functional as Fun

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


