import os
import json
import sys

from text_dl.models import model_factory
from text_dl.training import trainer_factory, optimizer_factory
from text_dl.generators import generators_factory

def main():
    '''
    Trains the model using the configurations
    '''
    with open(sys.argv[1]) as f:
        conf = json.load(f)
    
    model_config = conf['model']
    trainer_config = conf['trainer']
    generator_config = conf['generator']
    optimizer_config = conf['optimizer']
    
    # Preprocessor object
    vocabulary, train_itr, val_itr = generators_factory(model_config, generator_config)

    # Model object
    model = model_factory(model_config, vocabulary)

    # Trainer's function
    trainer = trainer_factory(trainer_config)

    optimizer = optimizer_factory(optimizer_config)

    # Results of the training
    results = trainer(model, optimizer, train_itr, val_itr)

    # TODO: Save results object somewhere.

if __name__ == "__main__":
    main()