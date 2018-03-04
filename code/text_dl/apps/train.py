import os
import json
import sys

from text_dl.modules.embeddings import embedding_factory
from text_dl.factories import generators_factory, trainer_factory, model_factory
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
    embeddings = embedding_factory(vocabulary, model_config['train_embedding'])
    model_config['embeddings'] = embeddings
    del model_config['train_embedding']
    model = model_factory(model_config)
    if use_cuda:
        model = model.cuda()

    # Trainer's function
    print("Creating trainer")
    trainer = trainer_factory(trainer_config)

    # Results of the training
    print("Training...")
    trainer.train(model, train_itr, val_itr)

if __name__ == "__main__":
    main()