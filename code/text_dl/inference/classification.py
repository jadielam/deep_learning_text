import torch
import torchtext.data as data

def extra_vocabs_fields_generator(training_path, batch_size = 1, 
                                    validation_path = None, 
                                    vocab_type = "glove.twitter.27B.200d", 
                                    entities_annotations = ['B-PARK', 'I-PARK', 'B-ATTR', 'I-ATTR', 'B-REST', 'I-REST', 'B-RESO', 'I-RESO', 'B-PLAC', 'I-PLAC', 'B-CHAR', 'I-CHAR', 'B-ENTE', 'I-ENTE']):
    text_field = data.Field(sequential = True, eos_token = '<eos>',
                            init_token = '<sos>', pad_token = '<pad>',
                            unk_token = '<unk>')
    
    annotations_field = data.Field(sequential = True, eos_token = '<eos>',
                                    init_token = '<sos>', pad_token = '<pad>',
                                    unk_token = '<unk>')
    
    dataset = data.Dataset(examples = [], fields = [("text", text_field), ('ner', annotations_field)])
    vocabulary_datasets = [data.TabularDataset(path = a, format = "csv",
                                                fields = [("text", text_field), ('ner', annotations_field)],
                                                skip_header = True) if a else None for a in 
                                                [training_path, validation_path]]
    text_field.build_vocab(*[a for a in vocabulary_datasets if a], vectors = vocab_type)
    annotations_field.build_vocab(*[a for a in vocabulary_datasets if a], 
                                    vectors = OneHotEncoderVectors(entities_annotations))

    text_vocabulary = text_field.vocab
    ner_vocabulary = annotations_field.vocab
    return text_vocabulary, ner_vocabulary, ('text', text_field), ('ner', annotations_field), dataset

def fields_generator(training_path, batch_size = 1, validation_path = None, vocab_type = "glove.twitter.27B.200d"):
    text_field = data.Field(sequential = True, eos_token = '<eos>',
                            init_token = '<sos>', pad_token = '<pad>',
                            unk_token = '<unk>')
    dataset = data.Dataset(examples = [], fields = [("text", text_field)])
    vocabulary_datasets = [data.TabularDataset(path = a, format = "csv",
                                                fields = [("text", text_field)],
                                                skip_header = True) if a else None for a in 
                                                [training_path, validation_path]]
    text_field.build_vocab(*[a for a in vocabulary_datasets if a], vectors = vocab_type)
    vocabulary = text_field.vocab
    return vocabulary, ('text', text_field), dataset