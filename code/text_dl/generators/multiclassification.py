import torchtext.data as data

def csv_generator(training_path, validation_path, nb_classes, batch_size):
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
    label_fields = [data.Field(sequential = False, use_vocab = False) for _ in range(nb_classes)]
    fields = [("id", None), ("text", text_field)] + [("label_{}".format(i), label_fields[i]) 
             for i in range(len(label_fields))]     
    
    datasets = [data.TabularDataset(path = a, format = "csv", 
                                    fields = fields, skip_header = False) if a else None for a in 
                                    [training_path, validation_path]]
    text_field.build_vocab(*[a for a in datasets if a], vectors = "glove.twitter.27B.200d")
    vocabulary = text_field.vocab
    [training_itr, validation_itr] = [data.Iterator(dts, batch_size = batch_size, 
                                        device = -1) if dts else None for dts in datasets]
    return vocabulary, training_itr, validation_itr