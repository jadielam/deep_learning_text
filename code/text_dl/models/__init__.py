'''
Contains model definitions for the different kinds of problems to solve.
'''
from text_dl.common.factories import generic_factory
from text_dl.models.multiclassification.simple_multiclassification import SimpleMulticlassificationModel
from text_dl.models.multiclassification.custom_multiclassification import CustomMulticlassificationModel
from text_dl.models.multiclassification.multiclassification import MulticlassificationModel

from text_dl.models.classification.classification import ClassificationModel
from text_dl.models.classification.attention_classification import AttentionClassificationModel

MODELS_D = {
    "CustomMulticlassificationModel": CustomMulticlassificationModel,
    "SimpleMulticlassificationModel": SimpleMulticlassificationModel,
    "MulticlassificationModel": MulticlassificationModel,
    "ClassificationModel": ClassificationModel,
    "AttentionClassificationModel": AttentionClassificationModel
}

models_factory = generic_factory(MODELS_D, "model")