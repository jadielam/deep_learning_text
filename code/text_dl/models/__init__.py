'''
Contains model definitions for the different kinds of problems to solve.
'''
from text_dl.common.factories import generic_factory
from text_dl.models.multiclassification.simple_multiclassification import SimpleMulticlassificationModel
from text_dl.models.multiclassification.custom_multiclassification import CustomMulticlassificationModel
from text_dl.models.multiclassification.multiclassification import MulticlassificationModel

MODELS_D = {
    "CustomMulticlassificationModel": CustomMulticlassificationModel,
    "SimpleMulticlassificationModel": SimpleMulticlassificationModel,
    "MulticlassificationModel": MulticlassificationModel
}

models_factory = generic_factory(MODELS_D, "model")