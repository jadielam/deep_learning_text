import torchtext.data as data

def csv_generator(csv_folder, train_suffix, val_suffix, nb_classes, batch_size):
    '''
    csv_folder: str
    train_suffix: str
    val_suffix: str
    nb_classes: int
    batch_size: int
    '''
    text_field = data.Field(sequential = True, eos_token = '<eos>',
                            init_token = '<sos>', pad_token = '<pad>',
                            unk_token = '<unk>')
    label_fields = [data.Field(sequential = False) for _ in range(nb_classes)]
    fields = [("id", None), ("text", text_field)] + 
                [("label_{}").format(i), label_fields[i]) for in range(len(label_fields))]
    
    training, validation = data.TabularDataset(path = csv_path, train = train_suffix,
                                    validation = val_suffix, 
                                    format = "csv", fields = fields)
    text_field.build_vocab(training, validation)
    [training_itr, validation_itr] = [data.Iterator(dts, batch_size = batch_size, 
                                        device = -1) for dts in [training, validation]]
    return training_itr, validation_itr