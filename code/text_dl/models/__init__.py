'''
Contains model definitions for the different kinds of problems to solve.
'''


from text_dl.modules.embeddings import embedding_factory
from text_dl.models.multiclassification.simple_multiclassification import SimpleMulticlassificationModel

def model_factory(conf, vocab):
    '''
    Returns a model built according to the conf parameters.
    
    Arguments:
    - conf (dict): model configuration
    - vocab (:obj:`torchtext.vocab.Vocab`): Vocabulary to be used for embeddings.
    '''
    
    if conf['type'] == 'multiclassification':
        nb_classes = conf['params']['nb_classes']
        train_embedding = conf['params']['train_embedding']
        batch_size = conf['params']['batch_size']
        max_sequence_length = conf['params']['max_sequence_length']
        embeddings = embedding_factory(vocab, train_embedding)
        model = SimpleMulticlassificationModel(embeddings, batch_size, 
                                            nb_classes, max_sequence_length)
        return model
    else:
        raise ValueError("Incorrect model type: {}".format(conf['type']))
    