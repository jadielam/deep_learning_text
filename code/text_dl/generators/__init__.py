from text_dl.common.factories import generic_factory
from .multiclassification import csv_generator as multi_csv_generator

GENERATORS_D = {
    "multiclass": multi_csv_generator
}

generators_factory = generic_factory(GENERATORS_D, "generator")