from functools import partial
import torch
from torch import optim
from sklearn.metrics import confusion_matrix
import numpy as np

from text_dl.common.factories import generic_factory

from text_dl.training.statistics import Statistics
from text_dl.training.callbacks import PrintCallback, ModelSaveCallback, HistorySaveCallback


def evaluate(model, val_itr, input_transform_f):
    '''
    Returns the total loss of the model on the validation and a confusion matrix
    dataset
    '''
    if val_itr is None:
        return None

    total_loss = 0.0    
    model.eval()

    #1. Calculating the loss
    for _, batch in enumerate(val_itr):
        loss = model.loss(input_transform_f(batch), batch.target)
        total_loss += loss.data.item()
    
    #2. Calculate accuracy
    model.train()
    return total_loss / len(val_itr)

class Trainer:
    def __init__(self, nb_epochs = 10, optimizer = {"type": "adam"}, callbacks = None, scheduler = None,
                model_weights_path = None, input_transform_f = lambda x : x):
        self.nb_epochs = nb_epochs
        self.optimizer_factory_conf = optimizer
        self.scheduler_factory_conf = scheduler

        self.callbacks = set()
        if callbacks is not None:
            self.callbacks = set([callbacks_factory(conf) for conf in callbacks])
        self.callbacks.add(PrintCallback())
        self.callbacks.add(ModelSaveCallback())
        self.callbacks.add(HistorySaveCallback())

        self.model_weights_path = model_weights_path
        self.input_transform_f = input_transform_f

    def train(self, model, train_itr, val_itr):
        '''
        Trains a model for self.nb_epochs, using the data given by
        train_itr, and evaluating val_itr

        Arguments:
            - model (::obj::`text_dl.models.model.Model`)
            - train_itr (::obj)

        '''

        #0. Load weights if present
        if self.model_weights_path is not None:
            model_weights = torch.load(self.model_weights_path)
            model.load_state_dict(model_weights)

        #1. Build optimizer
        self.optimizer_factory_conf['params']['params'] = model.trainable_parameters()
        optimizer = optimizers_factory(self.optimizer_factory_conf)

        #2. Build scheduler
        if self.scheduler_factory_conf is not None:
            self.scheduler_factory_conf['params']['optimizer'] = optimizer
            scheduler = schedulers_factory(self.scheduler_factory_conf)
        else:
            scheduler = None

        #3. Starting the training process
        epoch_stats = Statistics(self.nb_epochs)
        for epoch_idx in range(self.nb_epochs):
            
            for callback in self.callbacks:
                callback.on_epoch_begin(epoch_idx, model, epoch_stats)

            total_loss_value = 0.0
            iter_stats = Statistics(len(train_itr))
            for iter_idx in range(len(train_itr)):
                for callback in self.callbacks:
                    callback.on_iter_begin(iter_idx, epoch_idx, model, iter_stats)

                batch = next(train_itr.__iter__())
                loss = model.loss(self.input_transform_f(batch), batch.target)
                total_loss_value += loss.data.item()

                #Update iteration statistics and gradients
                iter_stats.update_stat("train_loss", total_loss_value / (iter_idx + 1))
                iter_stats.step()
                loss.backward()
                optimizer.step()

                #Make call to callbacks
                for callback in self.callbacks:
                    callback.on_iter_end(iter_idx, epoch_idx, model, iter_stats)
            
            #Update epoch statistics
            val_loss = evaluate(model, val_itr, self.input_transform_f)
            epoch_stats.update_stat("train_loss", total_loss_value / len(train_itr))
            epoch_stats.update_stat("val_loss", val_loss)
            epoch_stats.step()

            if scheduler is not None:
                scheduler.step()
            
            #Make call to callbacks
            for callback in self.callbacks:
                callback.on_epoch_end(epoch_idx, model, epoch_stats)


TRAINERS_D = {
    "simple": Trainer
}

OPTIMIZERS_D = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "adadelta": optim.Adadelta,
    "adagrad": optim.Adagrad,
    "rmsprop": optim.RMSprop
}

SCHEDULERS_D = {
    "lambda": optim.lr_scheduler.LambdaLR,
    "step": optim.lr_scheduler.StepLR,
    "multistep": optim.lr_scheduler.MultiStepLR,
    "exponential": optim.lr_scheduler.ExponentialLR
}

CALLBACKS_D = {
    "save_history": callbacks.HistorySaveCallback,
    "save_model": callbacks.ModelSaveCallback
}

trainers_factory = generic_factory(TRAINERS_D, "trainer")
optimizers_factory = generic_factory(OPTIMIZERS_D, "optimizer")
schedulers_factory = generic_factory(SCHEDULERS_D, "scheduler")
callbacks_factory = generic_factory(CALLBACKS_D, "callback")

