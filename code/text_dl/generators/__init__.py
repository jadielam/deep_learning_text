from text_dl.common.factories import generic_factory
import text_dl.generators.multiclassification

GENERATORS_D = {
    "multiclass": text_dl.generators.multiclassification.csv_generator
}

generators_factory = generic_factory(GENERATORS_D, "generator")