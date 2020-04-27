import torch


class CPCLoss(torch.nn.Module):

    def __init__(self, args):
        super().__init__()

        self.tau = args.temperature
        self.device = args.device

    def forward(self, x, y_hat, y):
        """
        Input: x, y are matrices with shape (batch, hid_size)
        Return: NTxent loss
        """

        print(x.shape, y_hat.shape, y.shape)

        x = x.view(x.shape[0], -1).to(self.device)
        y_hat = y_hat.view(y_hat.shape[0], -1).to(self.device)
        y = y.view(y.shape[0], -1).to(self.device)

        x_norm = x / (torch.norm(x, dim=1).reshape(-1, 1))
        y_norm = y / (torch.norm(y, dim=1).reshape(-1, 1))
        y_hat_norm = y_hat / (torch.norm(y_hat, dim=1).reshape(-1, 1))
        yhy_norm = torch.cat([y_hat_norm, y_norm], dim=0)
        yyh_norm = torch.cat([y_norm, y_hat_norm], dim=0)
        xyhy_norm = torch.cat([x_norm, y_hat_norm, y_norm], dim=0)

        sim_mat_1 = xyhy_norm @ xyhy_norm.T
        exp_mat = torch.exp(sim_mat_1 / self.tau)
        exp_mat_row_sum = torch.sum(exp_mat, dim=1)
        exp_mat_diag = torch.diag(exp_mat)
        denominator = (exp_mat_row_sum - exp_mat_diag)[3:]

        sim_mat_2 = yhy_norm @ yyh_norm.T
        numerator = torch.exp(torch.diag(sim_mat_2) / self.tau)
        print(numerator.shape, denominator.shape)
        nt_xent = - torch.log(numerator / denominator)
        return torch.mean(nt_xent)


class CPCLoss_alter(torch.nn.Module):

    def __init__(self, args):
        super().__init__()

        self.tau = args.temperature
        self.device = args.device

    def forward(self, x, y):
        """
        Input: x, y are matrices with shape (batch, hid_size)
        Return: NTxent loss
        """
        x = x.view(x.shape[0], -1).to(self.device)
        y = y.view(y.shape[0], -1).to(self.device)
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
