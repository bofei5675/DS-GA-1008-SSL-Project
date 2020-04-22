import torch

class CPCLoss(torch.nn.Module):

    def __init__(self, args):
        super().__init__()

        self.device = args.device
        self.use_cosine = args.use_cosine_similarity
        self.loss_function = self._get_similarity_function(
            args.use_cosine_similarity)
        self.embed_size = args.embed_size

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_similarity
        else:
            return self._dot_simililarity

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.view(-1, self.embed_size),
                            y.view(-1, self.embed_size), dims=2).to(self.device)
        return v

    def forward(self, x, y):
        return torch.sum(self.loss_function(x, y))