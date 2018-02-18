from text_dl.generators.multiclassification import csv_generator as multi_csv_generator

def generators_factory(conf):
    if conf['type'] == 'multiclassification':
        return multi_csv_generator(conf['params']['path'], 
                                    conf['params']['training_file'],
                                    conf['params'].get('validation_file', None),
                                    conf['params']['nb_classes'],
                                    conf['params']['batch_size'])
                    
    else:
        raise ValueError("Incorrect generator type: {}".format(conf['type']))