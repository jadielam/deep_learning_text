import os
import json
import sys
import torch

from text_dl.modules.embeddings import MultiEmbedding
from text_dl.models import models_factory
from text_dl.training import trainers_factory
from text_dl.generators import generators_factory
from text_dl.common.devices import use_cuda

def embedding_factory(vocabularies_l):
    vocab_sizes = [vocab.vectors.shape[0] for vocab in vocabularies_l]
    hidden_sizes = [vocab.vectors.shape[1] for vocab in vocabularies_l]
    vectors = [vocab.vectors for vocab in vocabularies_l]
    are_trainable = [False for vocab in vocabularies_l]

    return MultiEmbedding(vocab_sizes, hidden_sizes, vectors, are_trainable)

def pad_batches(batches):

    # assuming trailing dimensions and type of all the Variables
    # in sequences are same and fetching those from sequences[0]
    max_seq_len = 0
    for _, batch in enumerate(batches):
        if max_seq_len < batch.size()[0]:
            max_seq_len = batch.size()[0]
    
    out_batches = []
    
    for _, batch in enumerate(batches):
        out_dims = batch.size()
        length = out_dims[0]
        out_dims[0] = max_seq_len
        out_batch = torch.zeros(*out_dims)
        out_batch[:length, :] = batch
        out_batches.append(out_batch)

    return out_batches

def input_transform_f(batch):
    [batch.text, batch.ner] = pad_batches([batch.text, batch.ner])
    return torch.stack([batch.text, batch.ner], 2)

def main():
    '''
    Trains the model using the configurations
    '''
    with open(sys.argv[1]) as f:
        conf = json.load(f)
    
    model_config = conf['model']
    trainer_config = conf['trainer']
    generator_config = conf['generator']
    
    # Preprocessor object
    print("Creating vocabulary and iterators")
    text_vocabulary, annotation_vocabulary, train_itr, val_itr = generators_factory(generator_config)

    # Model object
    print("Creating model")
    
    # TODO: This is the only part of the code that is not pure enough.
    # Otherwise, all the factories look nice to me.
    embedding = embedding_factory([text_vocabulary, annotation_vocabulary])
    
    model_config['params']['embedding'] = embedding
    del model_config['params']['train_embedding']
    model = models_factory(model_config)
    if use_cuda:
        model = model.cuda()

    # Trainer's function
    print("Creating trainer")
    trainer_config['input_transform_f'] = input_transform_f
    trainer = trainers_factory(trainer_config)

    # Results of the training
    print("Training...")
    trainer.train(model, train_itr, val_itr)


if __name__ == "__main__":
    main()