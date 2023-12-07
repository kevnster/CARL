# -- IMPORTS --
import os, sys, random, gc, torch, warnings, tqdm

import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as u
import torch.multiprocessing as mp

from ADAM import CustomizedAdam
from CBAM_VAE import AttentionVAE
from typing import Tuple

# -- SET-UP --
torch.cuda.empty_cache()
gc.collect()
try:
     mp.set_start_method('spawn')
except RuntimeError:
    pass
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if not torch.cuda.is_available():
    warnings.warn('You are using CPU, this will be very slow, consider using GPU.')

def encoder_setup(vae: bool, weight_dir: str, z_dim: int, print_arch: bool = False) -> Tuple[nn.Module, str]:
    """
    Setup the encoder network.

    :param vae: True for VAE, False for CNN encoder.
    :param weight_dir: Directory of the encoder weights.
    :param z_dim: Dimension of the latent space.
    :param print_arch: True to print the architecture of the encoder.

    :return: Tuple containing the encoder network and the name of the encoder.
    """
    if vae:
        encoder_net = AttentionVAE(z_dim)
        model_filename = 'VAE'
    else:
        encoder_net = nn.CNN_Encoder()
        model_filename = 'Encoder'

    encoder_net.to(DEVICE)

    # Load the state dict with appropriate device mapping
    state_dict = torch.load(weight_dir, map_location=DEVICE)
    encoder_net.load_state_dict(state_dict)

    if vae:
        encoder_net.eval()  # Set to eval mode for VAE
        for param in encoder_net.parameters():
            param.requires_grad = False
    else:
        encoder_net.train()  # Set to train mode for CNN Encoder
        encoder_net.share_memory()  # Useful for multiprocessing

    if print_arch:
        print(encoder_net)

    return encoder_net, model_filename

class Global_Net(nn.Module):
    """
    Global network for the A3C algorithm.

    Attributes:
        action_dim (int): Dimension of the action space.
        z_dim (int): Dimension of the latent space.
        upscale_dim (int): Dimension of the upscale layer.
        mid1 (int): Dimension of the first middle layer.
        mid2 (int): Dimension of the second middle layer.
        mid3 (int): Dimension of the third middle layer.
        n_lstm (int): Number of LSTM layers.
        p (float): Dropout probability.
    """
    def __init__(self, action_dim, z_dim=64, upscale_dim=8, mid1=256, mid2=128, mid3=64, n_lstm=2, p=0):
        super(Global_Net, self).__init__()
        self._init_layers(action_dim, z_dim, upscale_dim, mid1, mid2, mid3, n_lstm, p)

    def _init_layers(self, action_dim, z_dim, upscale_dim, mid1, mid2, mid3, n_lstm, p):
        # Upscale layer
        self.upscale_layer = nn.Sequential(
            nn.Linear(action_dim + 1, 2 * upscale_dim),
            nn.ReLU6(),
            nn.Linear(2 * upscale_dim, 2 * upscale_dim),
            nn.ReLU6(),
            nn.Linear(2 * upscale_dim, max(upscale_dim, action_dim + 1)),
            nn.ReLU6(),
        )
        self._init_layer(self.upscale_layer)

        # LSTM and associated layers
        lstm_input_dim = 2 * z_dim + max(upscale_dim, action_dim + 1)
        if n_lstm != 0:
            self.lstm = nn.LSTM(lstm_input_dim, mid1, n_lstm, batch_first=True, dropout=p)
            self.before_lstm = nn.Sequential(nn.BatchNorm1d(lstm_input_dim), nn.ReLU6())
            self.after_lstm = nn.Sequential(nn.BatchNorm1d(mid1), nn.ReLU6())
            self._init_layer(self.before_lstm)
            self._init_layer(self.after_lstm)
        else:
            mid1 = lstm_input_dim

        # First layer
        self.first_layer = nn.Sequential(nn.Linear(mid1, mid2), nn.ReLU6(), nn.Linear(mid2, mid3), nn.ReLU6())
        self._init_layer(self.first_layer)

        # Actor and Critic
        self.actor = nn.Linear(mid3, action_dim)
        self.critic = nn.Linear(mid3, 1)
        self._init_layer(self.actor)
        self._init_layer(self.critic)

        # Other attributes
        self.distribution = nn.distributions.Categorical
        self.entropy_coeff = 0.01

    def _init_layer(self, layer):
        """Initialize a layer or sequence of layers."""
        for m in layer.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, 0)

    def forward(self, cur_z, target_z, pre_act, hx, cx) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass of the global network."""
        # Upscale previous action if needed
        act_map = self.upscale_layer(pre_act) if self.upscale_dim > self.action_dim + 1 else pre_act

        # Concatenate current and target latent states with the action map
        x = torch.cat((cur_z, target_z, act_map), dim=1)

        # Pass through LSTM if present
        if self.n_lstm != 0:
            x, (hx, cx) = self.lstm(x.unsqueeze(0), (hx.detach(), cx.detach()))
            x = x.squeeze(0)

        # Pass through the first layer
        x = self.first_layer(x)

        # Actor and Critic computations
        logits = self.actor(x)
        values = self.critic(x)

        return logits, values, (hx, cx)

    def choose_act(self, cur_z, target_z, pre_act, hx, cx) -> Tuple[int, Tuple[torch.Tensor, torch.Tensor]]:
        """Choose an action based on the current state."""
        self.eval()
        logits, _, (hx, cx) = self.forward(cur_z, target_z, pre_act, hx, cx)
        probs = F.softmax(logits, dim=-1)
        action = probs.multinomial(1).item()
        return action, (hx, cx)

    def loss_func(self, cur_z, target_z, pre_a, hx, cx, action, v_t) -> torch.Tensor:
        """Calculate the loss for training the global network."""
        self.train()
        logits, values, _ = self.forward(cur_z, target_z, pre_a, hx, cx)
        td = v_t - values
        c_loss = td.pow(2)
        probs = F.softmax(logits, dim=-1)
        m = self.distribution(probs)
        action_loss = -(m.log_prob(action) * td.detach().squeeze())
        total_loss = (c_loss + action_loss).mean()
        return total_loss


class Worker(mp.Process):
    """
    Local worker for asynchronous training in A3C algorithm.

    Attributes:
        name (str): Name of the worker.
        cur_world (str): Identifier of the current world.
        g_ep (mp.Value): Global episode counter.
        res_queue (mp.Queue): Queue to push results.
        action_dim (int): Dimension of action space.
        global_net (nn.Module): Global network.
        local_net (nn.Module): Local network.
        encoder_net (nn.Module): Encoder network.
        optimizer (torch.optim.Optimizer): Optimizer.
        clip_grad (float): Gradient clipping value.
        lock (mp.Lock): Multiprocessing lock.
        length_limit (int): Length limit of an episode.
        max_step (int): Max step per episode.
        automax_step (int): Max steps for automatic mode.
        backprop_iter (int): Backpropagation iterations.
        gamma (float): Discount factor.
    """

    def __init__(self, action_dim, max_glob_ep, input_dim, length_limit, max_step, automax_step, backprop_iter, gamma,
                 global_net, opt,
                 encoder_net, global_ep, res_queue, lock, name):
        super(Worker, self).__init__()
        self.name = f'W{name}'
        self.cur_world = 'FloorPlan303'

        self.g_ep, self.res_queue = global_ep, res_queue
        self.action_dim = action_dim

        self.global_net = global_net.to(DEVICE)
        self.local_net = Global_Net(action_dim=action_dim).to(DEVICE)
        self.encoder_net = encoder_net.to(DEVICE)
        self.clip_grad = 0.1
        self.optimizer = opt
        self.gamma = gamma

        self.max_glob_ep = max_glob_ep
        self.max_step = max_step
        self.automax_step = automax_step
        self.input_dim = input_dim
        self.length_limit = length_limit
        self.backprop_iter = backprop_iter
        self.lock = lock

        self.init_memory_buffers()
        self.init_lstm_states(action_dim)

    def init_memory_buffers(self):
        """Initialize memory buffers for the worker."""
        self.memory_cur_z, self.memory_target_z = [], []
        self.memory_actions, self.memory_pre_actions, self.memory_rewards = [], [], []
        self.buffer_cur_z, self.buffer_target_z = [], []
        self.buffer_actions, self.buffer_pre_actions, self.buffer_rewards = [], [], []
        self.target_values = []

    def init_lstm_states(self, action_dim):
        """Initialize LSTM states."""
        self.hx = torch.zeros(self.local_net.n_lstm, self.local_net.mid1, device=DEVICE,
                              dtype=torch.float32) if self.local_net.n_lstm != 0 else 0
        self.cx = torch.zeros(self.local_net.n_lstm, self.local_net.mid1, device=DEVICE,
                              dtype=torch.float32) if self.local_net.n_lstm != 0 else 0
        self.pre_action = torch.zeros(1, action_dim + 1, device=DEVICE, dtype=torch.float32)

    def mem_enchance(self) -> None:
        """Extend the memory of the local A2C."""
        self.memory_cur_z = self.buffer_cur_z + self.memory_cur_z
        self.memory_target_z = self.buffer_target_z + self.memory_target_z
        self.memory_actions = self.buffer_actions + self.memory_actions
        self.memory_pre_actions = self.buffer_pre_actions + self.memory_pre_actions
        self.memory_rewards = self.buffer_rewards + self.memory_rewards

    def push_and_pull(self) -> None:
        """Push the local network to the global network and pull the global network to the local network."""
        # Reverse rewards and compute target values
        self.target_values = [self.gamma * val for val in self.memory_rewards[::-1]]
        self.target_values.reverse()

        # Compute the loss and backpropagate
        self.optimizer.zero_grad()
        loss = self.local_net.loss_func(
            torch.vstack(self.memory_cur_z),
            torch.vstack(self.memory_target_z),
            torch.vstack(self.memory_pre_actions),
            self.hx,
            self.cx,
            torch.tensor(self.memory_actions, device=DEVICE),
            torch.tensor(self.target_values, device=DEVICE)[:, None])
        loss.backward()

        # Synchronize the local network with the global network
        for local_param, global_param in zip(self.local_net.parameters(), self.global_net.parameters()):
            global_param._grad = local_param.grad
        self.optimizer.step()
        self.local_net.load_state_dict(self.global_net.state_dict())

    def run(self) -> None:
        """Run the worker for training with individual simulation environments."""
        # self.env = ActivateEnv(self.action_space, self.length_limit, self.max_step, self.automax_step)
        self.reset_episode_stats()

        while self.g_ep.value < self.max_glob_ep:
            self.save_global_net_if_needed()

            self.cur_frame, self.target_frame, self.shortest_len, self.target_name = self.env.reset()
            self.target_obs = self.encoder_net.encoding_fn(self.target_frame)

            while not self.is_episode_finished():
                self.cur_obs = self.encoder_net.encoding_fn(self.cur_frame)
                self.action, (self.hx, self.cx) = self.local_net.choose_act(self.cur_obs, self.target_obs,
                                                                            self.pre_action, self.cx, self.hx)
                self.cur_frame, self.collided, self.step_reward, self.done, self.succeed = self.env.step(self.action)

                self.update_episode_stats()
                self.store_experience(self.action, self.step_reward)

                if self.should_backpropagate():
                    self.push_and_pull()
                    self.mem_clean()

                if self.done:
                    self.log_episode_results()
                    break

                self.update_pre_action()
        if not self.flag:
            self.res_queue.put([None, None, None, None])

        def reset_episode_stats(self):
            """Reset statistics for the new episode."""
            self.flag = False
            self.ep_spl = 0.
            self.episode_steps = 1
            self.ep_reward = 0.
            self.episode_collides = 0
            self.collided = False
            self.reset_lstm_states()

        def reset_lstm_states(self):
            """Reset LSTM states if applicable."""
            if self.local_net.n_lstm != 0:
                self.cx = torch.zeros_like(self.cx)
                self.hx = torch.zeros_like(self.hx)
            self.pre_action = torch.zeros_like(self.pre_action)

        def save_global_net_if_needed(self):
            """Save global network at regular intervals."""
            if self.g_ep.value % 100 == 0:
                torch.save(self.global_net.state_dict(), f'./temp_a3c_models/temp_a3c_model_E{self.g_ep.value}')

        def is_episode_finished(self) -> bool:
            """Check if the current episode is finished."""
            return self.g_ep.value >= self.max_glob_ep or self.done

        def update_episode_stats(self):
            """Update episode statistics after each step."""
            self.ep_reward += self.step_reward
            self.pre_action[:, self.action_dim] = int(self.collided)
            self.episode_collides += int(self.collided)

        def store_experience(self, action, reward):
            """Store experience in memory."""
            self.memory_actions.append(action)
            self.memory_pre_actions.append(self.pre_action.clone())
            self.memory_cur_z.append(self.cur_obs.clone())
            self.memory_target_z.append(self.target_obs.clone())
            self.memory_rewards.append(reward)

        def should_backpropagate(self) -> bool:
            """Check if it's time to backpropagate."""
            return self.episode_steps % self.backprop_iter == 0 or self.done

        def log_episode_results(self):
            """Log results of the episode and update global episode counter."""
            with self.lock:
                self.update_global_episode()
                self.record_episode_results()
                gc.collect()

        def update_global_episode(self):
            """Update the global episode counter."""
            self.g_ep.value += 1

        def record_episode_results(self):
            """Record results of the episode."""
            self.calculate_spl()
            self.res_queue.put([self.succeed, self.ep_reward, self.ep_spl, self.episode_collides])
            tqdm.tqdm.write(
                f'E{self.g_ep.value} - {self.name} - {self.env.max_step} | Succ:{self.succeed} | Coll:{self.episode_collides} | SPL:{self.ep_spl:.2f} | EpR:{self.ep_reward:.2f} | Target:{self.env.goal_name}')

        def calculate_spl(self):
            """Calculate Success weighted by Path Length (SPL) for the episode."""
            if self.succeed:
                self.ep_spl = self.shortest_len / max(self.episode_steps * 0.125, self.shortest_len)
            else:
                self.ep_spl = 0

        def update_pre_action(self):
            """Update previous action for the next step."""
            self.pre_action = torch.zeros_like(self.pre_action)
            self.pre_action[:, self.action] = 1
            self.episode_steps += 1