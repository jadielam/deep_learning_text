import os
import json
import sys

from text_dl.modules.embeddings import embedding_factory
from text_dl.models import models_factory
from text_dl.training import trainers_factory
from text_dl.generators import generators_factory
from text_dl.common.devices import use_cuda

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
    vocabulary, train_itr, val_itr = generators_factory(generator_config)

    # Model object
    print("Creating model")
    
    # TODO: This is the only part of the code that is not pure enough.
    # Otherwise, all the factories look nice to me.
    embedding = embedding_factory(vocabulary, model_config['params']['train_embedding'])
    model_config['params']['embedding'] = embedding
    del model_config['params']['train_embedding']
    model = models_factory(model_config)
    if use_cuda:
        model = model.cuda()

    # Trainer's function
    print("Creating trainer")
    trainer_config['input_transform_f'] = lambda x: x.text
    trainer = trainers_factory(trainer_config)

    # Results of the training
    print("Training...")
    trainer.train(model, train_itr, val_itr)


if __name__ == "__main__":
    main()