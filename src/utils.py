# -- IMPORT --
import torch, gc
import numpy as np
from torch import nn

def convert_tensor_to_rgb(input_tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to an RGB image represented as a NumPy array.

    :param input_tensor: A tensor to be converted.
    :return: NumPy array representing an RGB image.
    """
    image = input_tensor.cpu().clone()
    image = image.squeeze(0) if image.dim() > 3 else image
    image = nn.unloader(image)
    return np.array(image).transpose(1, 0, 2)

def numpy_to_tensor(array: np.ndarray, dtype: type = np.float32) -> torch.Tensor:
    """
    Convert a NumPy array to a PyTorch tensor.

    :param array: NumPy array to be converted.
    :param dtype: Target data type for the tensor.
    :return: PyTorch tensor.
    """
    if array.dtype != dtype:
        array = array.astype(dtype)
    return torch.from_numpy(array)

def initialize_layers(layers: list) -> None:
    """
    Initialize layers with a normal distribution for weights and zeros for biases.

    :param layers: A list of layers to be initialized.
    """
    for layer in layers:
        if hasattr(layer, 'weight'):
            nn.init.normal_(layer.weight, mean=0., std=0.1)
        if hasattr(layer, 'bias'):
            nn.init.constant_(layer.bias, 0.)

def sync_and_optimize(optimizer: torch.optim.Optimizer, local_model: nn.Module, global_model: nn.Module,
                      buffer_cur_z, buffer_target_z, buffer_pre_action, buffer_action, buffer_reward, gamma: float,
                      device: torch.device, hidden_state, cell_state) -> None:
    """
    Synchronize local model with the global model and perform optimization.

    :param optimizer: Optimizer for the models.
    :param local_model: Local model instance.
    :param global_model: Global model instance.
    :param buffer_cur_z: Buffer for current z values.
    :param buffer_target_z: Buffer for target z values.
    :param buffer_pre_action: Buffer for previous actions.
    :param buffer_action: Buffer for actions.
    :param buffer_reward: Buffer for rewards.
    :param gamma: Discount factor for future rewards.
    :param device: Device to perform computations on.
    :param hidden_state: Hidden state of the model.
    :param cell_state: Cell state of the model.
    """
    value = 0
    buffer_v_target = []
    for reward in reversed(buffer_reward):
        value = reward + gamma * value
        buffer_v_target.insert(0, value)

    loss = local_model.loss_function(
        torch.vstack(buffer_cur_z),
        torch.vstack(buffer_target_z),
        torch.vstack(buffer_pre_action),
        hidden_state,
        cell_state,
        torch.tensor(buffer_action, device=device),
        torch.tensor(buffer_v_target, device=device)[:, None]
    )

    optimizer.zero_grad()
    loss.backward()
    for local_param, global_param in zip(local_model.parameters(), global_model.parameters()):
        if local_param.grad is not None:
            global_param._grad = local_param.grad
    optimizer.step()

    local_model.load_state_dict(global_model.state_dict())
    gc.collect()

def log_episode_results(result_queue, episode_reward: float, success: bool, episode_spl: float) -> None:
    """
    Log the results of an episode.

    :param result_queue: Queue to store the results.
    :param episode_reward: Total reward for the episode.
    :param success: Whether the episode was successful.
    :param episode_spl: SPL (Success weighted by Path Length) metric for the episode.
    """
    result_queue.put([episode_reward, success, episode_spl])

def generate_model_filename(prefix: str = '', model_name: str = '', input_size: int = 0, max_global_episodes: int = 0,
                            self_adjusted_task_steps: int = None) -> str:
    """
    Generate a filename for saving the model.

    :param prefix: Prefix for the model filename.
    :param model_name: Name of the model.
    :param input_size: Input size of the model.
    :param max_global_episodes: Maximum number of global episodes.
    :param self_adjusted_task_steps: Number of self-adjusted task steps.
    :return: Generated model filename.
    """
    if model_name == 'Encoder':
        filename = f"A3C&{model_name}{input_size}-1"
        if self_adjusted_task_steps is not None:
            filename += f"_{self_adjusted_task_steps}N_ep{max_global_episodes}"
        else:
            filename += f"_ep{max_global_episodes}"
    else:
        filename = f"{prefix}A3C_{model_name}{input_size}-1"
        if self_adjusted_task_steps is not None:
            filename += f"_recon_{self_adjusted_task_steps}N_ep{max_global_episodes}"
        else:
            filename += f"_ep{max_global_episodes}"
    gc.collect()
    return filename

