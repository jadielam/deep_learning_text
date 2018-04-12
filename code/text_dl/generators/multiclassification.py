import torch
import torchtext.data as data
from text_dl.common.devices import use_cuda

#TODO: For now this is a hack to get this working quick.
# Later on, take a look at this sample code to figure out how
# to get this working in a less magical manner
# https://discuss.pytorch.org/t/how-to-do-multi-label-classification-with-torchtext/11571

def preprocessing_factory():
    '''
    Factory function that adds nb_classes as outside parameter
    '''
    def preprocessing(x):
        '''
        Converts a comma separated string of numbers into 
        a list of numbers
        '''
        entries = [float(a) for a in x.split(",")]
        return entries
    return preprocessing

def csv_generator(training_path, batch_size, validation_path = None, vocab_type = "glove.twitter.27B.200d"):
    '''
    Arguments:

    - training_path (str): path to the csv training file
    - batch_size (int): size of the batch
    - validation_path (str): path to the csv of the validation file
    - vocab_type (str): Indicates the pretrained word vectors type that will be used.
    '''
    text_field = data.Field(sequential = True, eos_token = '<eos>',
                            init_token = '<sos>', pad_token = '<pad>',
                            unk_token = '<unk>')
    target_field = data.Field(sequential = False, use_vocab = False, 
                                tensor_type = torch.FloatTensor,
                                preprocessing = preprocessing_factory())
    fields = [("id", None), ("text", text_field), ("target", target_field)]
    
    datasets = [data.TabularDataset(path = a, format = "csv", 
                                    fields = fields, skip_header = True) if a else None for a in 
                                    [training_path, validation_path]]
    text_field.build_vocab(*[a for a in datasets if a], vectors = vocab_type)
    vocabulary = text_field.vocab

    device_type = None if use_cuda else -1
    [training_itr, validation_itr] = [data.Iterator(dts, batch_size = batch_size, train = train,
                                        device = device_type, sort = False) if dts else None for (dts, train) in zip(datasets, [True, False])]
    return [vocabulary], training_itr, validation_itr