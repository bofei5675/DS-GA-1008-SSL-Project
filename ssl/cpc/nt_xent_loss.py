import torch

def nt_xent_loss(X, Y_hat, Y):
    '''
    Input: X, Y_hat, Y are matrices with shape (batch, seq, hid_size)
    Return: NT-Xent loss
    '''
    X = X.view(X.shape[0], -1)
    Y_hat = Y_hat.view(Y_hat.shape[0], -1)
    Y = Y.view(Y.shape[0], -1)

    X_norm = X / (torch.norm(X, dim=1).reshape(-1, 1))
    Y_norm = Y / (torch.norm(Y, dim=1).reshape(-1, 1))
    Y_hat_norm = Y_hat / (torch.norm(Y_hat, dim=1).reshape(-1, 1))
    YhY_norm = torch.cat([Y_hat_norm, Y_norm], dim=0)
    YYh_norm = torch.cat([Y_norm, Y_hat_norm], dim=0)
    XYhY_norm = torch.cat([X_norm, Y_hat_norm, Y_norm], dim=0)

    sim_mat_1 = XYhY_norm @ XYhY_norm.T
    exp_mat = torch.exp(sim_mat_1 / tau)
    exp_mat_row_sum = torch.sum(exp_mat, dim=1)
    exp_mat_diag = torch.diag(exp_mat)
    denominator = (exp_mat_row_sum - exp_mat_diag)[3:]

    sim_mat_2 = YhY_norm @ YYh_norm.T
    numerator = torch.exp(torch.diag(sim_mat_2) / tau)

    nt_xent = - torch.log(numerator / denominator)
    return torch.mean(nt_xent)
