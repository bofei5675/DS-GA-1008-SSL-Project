import torch

def rectangle(center, vector, l, w):
    '''
    Input: center, direction vector, length and width. Shape: 2, 2, 1, 1
    Output: Four corners of this rectangle
    '''
    vector = vector / torch.norm(vector)
    length = max([l, w])
    width = min([l, w])
    vector_p = torch.tensor([vector[1], -vector[0]])

    a = center + length * vector / 2 + width * vector_p / 2
    b = center + length * vector / 2 - width * vector_p / 2
    c = center - length * vector / 2 + width * vector_p / 2
    d = center - length * vector / 2 - width * vector_p / 2

    return a, b, c, d