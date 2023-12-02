import torch


class CustomizedAdam(torch.optim.Adam):
    """
    This is a customized Adam optimizer with shared memory for gradients.
    This optimizer is suitable for distributed training processes.
    """

    def __init__(self, params, lr=7e-4, betas=(0.92, 0.99), eps=1e-8, decay=0):
        super(CustomizedAdam, self).__init__(params, lr=lr, betas=betas, eps=eps, decay=decay)
        self.initialize()

    def initialize(self):
        for group in self.param_groups:
            for param in group['params']:
                optimizer_state = self.state[param]
                optimizer_state['step'] = 0
                optimizer_state['moving_avg'] = torch.zeros_like(param.data)
                optimizer_state['moving_avg_sq'] = torch.zeros_like(param.data)

                optimizer_state['moving_avg'].share_memory_()
                optimizer_state['moving_avg_sq'].share_memory_()
