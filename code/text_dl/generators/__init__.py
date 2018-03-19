from text_dl.common.factories import generic_factory
from .multiclassification import csv_generator as multiclass_csv_generator
from .classification import csv_generator as class_csv_generator

GENERATORS_D = {
    "multiclass": multiclass_csv_generator,
    "class": class_csv_generator
}

generators_factory = generic_factory(GENERATORS_D, "generator")