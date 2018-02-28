import torch
import torchtext.data as data
from text_dl.common.devices import use_cuda

def preprocessing_factory(nb_classes):
    '''
    Factory function that adds nb_classes as outside parameter
    '''
    def preprocessing(x):
        '''
        Converts a comma separated string of numbers into 
        a list of numbers
        '''
        entries = [float(a) for a in x.split(",")]
        if len(entries) != nb_classes:
            raise ValueError("The number of classes is not correct")
        return entries
    
    return preprocessing

def csv_generator(training_path, validation_path, nb_classes, batch_size, vocab_type = "glove.twitter.27B.200d"):
    '''
    csv_folder (str): Folder that contains the csv file
    train_suffix (str): Suffix of the csv files with the training data
    val_suffix (str): Suffix of the csv files with the validation data
    nb_classes (int): Number of classes in the classification
    batch_size (int): Size of the batch for the iterator
    '''
    text_field = data.Field(sequential = True, eos_token = '<eos>',
                            init_token = '<sos>', pad_token = '<pad>',
                            unk_token = '<unk>')
    target_field = data.Field(sequential = False, use_vocab = False, 
                                tensor_type = torch.FloatTensor,
                                preprocessing = preprocessing_factory(nb_classes))
    fields = [("id", None), ("text", text_field), ("target", target_field)]
    
    datasets = [data.TabularDataset(path = a, format = "csv", 
                                    fields = fields, skip_header = True) if a else None for a in 
                                    [training_path, validation_path]]
    text_field.build_vocab(*[a for a in datasets if a], vectors = vocab_type)
    vocabulary = text_field.vocab

    device_type = None if use_cuda else -1
    [training_itr, validation_itr] = [data.Iterator(dts, batch_size = batch_size, train = train,
                                        device = device_type) if dts else None for (dts, train) in zip(datasets, [True, False])]
    return vocabulary, training_itr, validation_itr