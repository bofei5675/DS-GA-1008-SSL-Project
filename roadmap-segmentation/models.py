import torch


class UNet(torch.nn.Modules):
    '''
    Create a UNet model
    Inputs: an image x with size [B, 3, H, W]
    Outputs: a featuren map y=model(x) with size [B, H / S, W / S]. Map this value to prob=torch.sigmoid(y)
    where S is a stride.
    The Label will be a 0-1 matrix with size y_true=[B, H/S, W/S]
    Loss for a specific instance in Batch:
    loss_b = \sum_{h,w} -y_true[h, w]log(prob[h, 2]) - (1 - y_true[h, w])log(1 - prob[h, 2])
    '''
    def __init__(self):
        pass