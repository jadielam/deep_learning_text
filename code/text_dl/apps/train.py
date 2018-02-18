import os
import json
import sys

from text_dl.models import model_factory
from text_dl.training import trainer_factory
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
    
    # Model object
    model = model_factory(model_config)

    # Trainer's function
    trainer = trainer_factory(trainer_config)

    # Preprocessor object
    generator = generators_factory(generator_config)
    
    # Results of the training
    results = trainer(model, generator)

    # TODO: Save results object somewhere.

if __name__ == "__main__":
    main()