import os
import json
import sys

from text_dl.models import model_factory
from text_dl.training import trainer_factory, optimizer_factory
from text_dl.factories import generators_factory
from text_dl.factories import trainer_factory
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
    optimizer_config = conf['trainer']['optimizer']
    
    # Preprocessor object
    print("Creating vocabulary and iterators")
    vocabulary, train_itr, val_itr = generators_factory(generator_config)

    # Model object
    print("Creating model")
    model = model_factory(model_config, vocabulary)
    if use_cuda:
        model = model.cuda()

    # Trainer's function
    print("Creating trainer")
    trainer = trainer_factory(trainer_config)

    print("Creating optimizer")
    #optimizer = optimizer_factory(optimizer_config, model)

    # Results of the training
    print("Training...")
    results = trainer.train(model, train_itr, val_itr)

    # TODO: Save results object somewhere.

    # TODO: Save serialized model and embeddings somewhere.
    # TODO: I should implement a method in model, called save.

if __name__ == "__main__":
    main()