import torch


class CPCLoss(torch.nn.Module):

    def __init__(self, args):
        super().__init__()

        self.tau = args.temperature

    def forward(self, x, y):
        """
        Input: x, y are matrices with shape (batch, hid_size)
        Return: NTxent loss
        """
        x = x.view(x.shape[0], -1)
        y = y.view(y.shape[0], -1)
        x_norm = x / (torch.norm(x, dim=1).reshape(-1, 1))
        y_norm = y / (torch.norm(y, dim=1).reshape(-1, 1))
        xy_norm = torch.cat([x_norm, y_norm], dim=0)
        yx_norm = torch.cat([y_norm, x_norm], dim=0)
    
        sim_mat_1 = xy_norm @ xy_norm.T
        exp_mat = torch.exp(sim_mat_1 / self.tau)
        exp_mat_row_sum = torch.sum(exp_mat, dim=1)
        exp_mat_diag = torch.diag(exp_mat)
        denominator = exp_mat_row_sum - exp_mat_diag

        sim_mat_2 = xy_norm @ yx_norm.T
        numerator = torch.exp(torch.diag(sim_mat_2) / self.tau)

        nt_xent = - torch.log(numerator / denominator)
        return torch.mean(nt_xent)
