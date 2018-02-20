from text_dl.generators.multiclassification import csv_generator as multi_csv_generator

def generators_factory(model_conf, generator_conf):
    '''
    Arguments:
        - model_conf (dict): model configuration
        - generator_conf (dict): generator configuration

    Returns:
        - vocab (:obj:`torchtext.vocab.Vocab`): vocabulary object or None if not necessary.
        - train_itr (:obj:`torchtext.data.iterator.Iterator`): training iterator
        - val_itr (:obj:`torchtext.data.iterator.Iterator`): validation iterator or None if not present
    '''
    if model_conf['type'] == 'multiclassification':
        return multi_csv_generator(generator_conf['params']['path'], 
                                    generator_conf['params']['training_file'],
                                    generator_conf['params'].get('validation_file', None),
                                    generator_conf['params']['nb_classes'],
                                    model_conf['params']['batch_size'])
                    
    else:
        raise ValueError("Incorrect generator type: {}".format(generator_conf['type']))