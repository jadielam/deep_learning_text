#1. Generators imports
import text_dl.generators.multiclassification

#2. Trainers imports
import text_dl.training

#3. Optimizers and schedulers imports
from torch import optim

#4. Models
import text_dl.models

#5. Callbacks
import text_dl.training.callbacks as callbacks

GENERATORS_D = {
    "multiclass": text_dl.generators.multiclassification.csv_generator
}

TRAINERS_D = {
    "simple": text_dl.training.Trainer
}

OPTIMIZERS_D = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "adadelta": optim.Adadelta,
    "adagrad": optim.Adagrad,
    "rmsprop": optim.RMSprop
}

SCHEDULERS_D = {
    "lambda": optim.lr_scheduler.LambdaLR,
    "step": optim.lr_scheduler.StepLR,
    "multistep": optim.lr_scheduler.MultiStepLR,
    "exponential": optim.lr_scheduler.ExponentialLR
}

CALLBACKS_D = {
    "save_history": callbacks.HistorySaveCallback,
    "save_model": callbacks.ModelSaveCallback
}

MODELS_D = {
    "CustomMulticlassificationModel": text_dl.models.multiclassification.custom_multiclassification.CustomMulticlassificationModel,
    "SimpleMulticlassificationModel": text_dl.models.multiclassification.simple_multiclassification.SimpleMulticlassificationModel
}

def generic_factory(type_factory_d, factory_type_s):
    '''
    Meta factory function

    Arguments:
        - type_factory_d (Dict[string -> callable])
        - factory_type_s (string)
    
    Returns:
        - factory (callable): the function to be used as factory
    '''
    def factory(conf):
        '''
        Given a configuration, returns a created object passing conf as
        parameter

        Arguments:
            - conf (dict): Configuration with parameters on it
        
        Returns:
            - object: Created object
        '''
        ftype = conf['type']
        params = conf.get("params", {})
        try:
            return type_factory_d[ftype](**params)
        except KeyError:
            raise ValueError("Incorrect {} type: {}".format(factory_type_s, ftype))

    return factory

models_factory = generic_factory(MODELS_D, "model")
generators_factory = generic_factory(GENERATORS_D, "generator")
trainers_factory = generic_factory(TRAINERS_D, "trainer")
optimizers_factory = generic_factory(OPTIMIZERS_D, "optimizer")
schedulers_factory = generic_factory(SCHEDULERS_D, "scheduler")
callbacks_factory = generic_factory(CALLBACKS_D, "callback")


