import torch
import torch.nn as nn
from text_dl.modules import Module

def embedding_factory(vocab, train_embedding = False):
    '''
    Creates an embedding given the vocabulary
    A vocabulary object is necessary
    
    Arguments:
    - vocab (:obj:`torchtext.vocab.Vocab`): Vocabulary to be used with the embeddings.
    - train_embedding (bool): `True` if embedding is to be trained, `False` otherwise

    Returns:
    - embedding (:obj:`torch.nn.Embedding`): The embedding layer to be used
    '''
    vocab_size = vocab.vectors.shape[0]
    hidden_size = vocab.vectors.shape[1]
    embedding = nn.Embedding(vocab_size, hidden_size)
    embedding.weight = nn.Parameter(vocab.vectors)
    embedding.weight.requires_grad = train_embedding
    return embedding

