import torch
import torch.nn as nn
from text_dl.modules import Module

from text_dl.modules.utils import Elementwise

class MultiEmbedding(nn.Module):
    def __init__(self, vocab_sizes, hidden_sizes, vectors, are_trainable, *args):
        '''
        Arguments:
            - vocab_sizes (list(int)): the vocabulary sizes of all
            the embeddings
            - hidden_sizes (list(int)): the hidden sizes (the number of features)
            for each of the embeddings
            - vectors: the vectors of each of the embeddings
            - are_trainable: list(boolean): indicating if the vectors of this 
            embedding are trainable or not.
        
        TODO: We need to assert that the len of all this lists is the same
        '''
        #1. Create the embeddings and load the vectors
        super(MultiEmbedding, self).__init__(*args)
        emb_params = zip(vocab_sizes, hidden_sizes)
        embeddings = [nn.Embedding(vocab_size, hidden_size) for vocab_size, hidden_size in emb_params]
        for idx, embedding in enumerate(embeddings):
            embedding.weight = nn.Parameter(vectors[idx])
            embedding.weight.requires_grad = are_trainable[idx]

        #2. Assign the embeddings to the elementwise layer
        self.elementwise_embeddings = Elementwise('concat', embeddings)
        self.embedding_dim = (sum(hidden_sizes))

    def forward(self, input_t):
        '''
        Arguments:
            - input_t (::obj::`torch.Tensor`): of (seq_len, batch, nb_embeddings)
        '''
        concat_embeddings = self.elementwise_embeddings(input_t)
        return concat_embeddings

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

