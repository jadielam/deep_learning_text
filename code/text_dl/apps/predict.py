import os
import json
import sys
import csv

import torch
import torchtext.data as data

from text_dl.inference import fields_factory
from text_dl.modules.embeddings import embedding_factory
from text_dl.models import models_factory
from text_dl.common.devices import use_cuda

#TODO: FOr now I will do this here. Later I will redraw it to
#something better.  Pytorch text is not designed to help you 
#out with the first pipeline for training and for testing
#so I need to change that.
def transform_query(query_text, name_field_t, dataset):
    #1. Create example from query text
    example = data.Example.fromlist([query_text], [name_field_t])
    examples = [example]

    #2. Create batch using example and dataset
    device_type = None if use_cuda else -1
    batch = data.Batch(data = examples, dataset = dataset, device = device_type)

    #3. Return batch
    return batch

def read_labels(labels_file_path):
    
    reader = csv.reader(open(labels_file_path, 'r'))
    labels_intent = {}
    for row in reader:
        key, value = row
        labels_intent[int(key)] = value
    
    return labels_intent

def main():
    with open(sys.argv[1]) as f:
        conf = json.load(f)
    
    model_weights_path = conf['model_weights_path']
    original_conf_file = conf['original_conf_file']
    labels_file_path = conf['labels_file_path']
    labels_intent = read_labels(labels_file_path)

    with open(original_conf_file) as f:
        original_conf = json.load(f)
    model_config = original_conf['model']
    generator_config = original_conf['generator']
    
    vocabulary, (name, text_field), dataset = fields_factory(generator_config)
    embedding = embedding_factory(vocabulary, train_embedding = False)
    model_config['params']['embedding'] = embedding
    del model_config['params']['train_embedding']

    model = models_factory(model_config)
    if use_cuda:
        model.cuda()
    model_weights = torch.load(model_weights_path)
    model.load_state_dict(model_weights)
    model.eval()

    query_text = None
    while True:
        query_text = input("Your question: ")
        if query_text == "exit":
            break
        batch = transform_query(query_text, (name, text_field), dataset)
        prediction = model.forward(batch.text)
        print(prediction)
        maximum = torch.max(prediction, 1)
        arg_max = maximum[1][0]
        print(labels_intent[arg_max])

if __name__ == "__main__":
    main()


    
    