from text_dl.common.factories import generic_factory
from .classification import fields_generator as class_fields_generator

FIELDS_D = {
    "class": class_fields_generator
}

fields_factory = generic_factory(FIELDS_D, "fields")