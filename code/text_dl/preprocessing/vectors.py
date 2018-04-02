import torch
from collections import defaultdict
import torchtext.data as data
import torchtext.vocab as vocab

class OneHotEncoderVectors(vocab.Vectors):
    def __init__(self, words):
        self.itos = {idx : word for idx, word in enumerate(words)}
        self.stoi = {word : idx for idx, word in enumerate(words)}
        self.vectors = torch.eye(len(words))
        self.dim = len(words)
        self.unk_init = torch.Tensor.zero_

