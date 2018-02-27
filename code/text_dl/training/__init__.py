from functools import partial
from torch import optim

def evaluate(model, val_itr):
    '''
    Returns the total loss of the model on the validation
    dataset
    '''
    if val_itr is None:
        return None

    total_loss = 0.0

    for _, batch in enumerate(val_itr):
        loss = model.loss(batch.text, batch.target)
        total_loss += loss.data[0]
    return total_loss / len(val_itr)

def train(model, optimizer, train_itr, val_itr, nb_epochs):
    '''
    Trains a model
    '''
    previous_epoch = 0
    total_loss_value = 0.0

    while True:
        batch = next(train_itr.__iter__())
        loss = model.loss(batch.text, batch.target)
        total_loss_value += loss.data[0]
        loss.backward()
        optimizer.step()

        if train_itr.epoch() == nb_epochs:
            break

        if train_itr.epoch() != previous_epoch:
            val_loss_value = evaluate(model, val_itr)
            tr_loss_value = total_loss_value / len(train_itr)
            total_loss_value = 0.0
            previous_epoch = train_itr.epoch()
                
            print("Epoch # {}: training loss - {} \t validation loss - {}".format(train_itr.epoch(), tr_loss_value, val_loss_value))

    val_loss_value = evaluate(model, val_itr)
    print("Epoch # {}: training loss - {} \t validation loss - {}".format(train_itr.epoch(), tr_loss_value, val_loss_value))


    
def trainer_factory(conf):
    nb_epochs = conf['params']['nb_epochs']
    train_f = partial(train, nb_epochs = nb_epochs)
    return train_f

def optimizer_factory(conf, model):
    '''
    For now, only return one kind of optimizer
    '''
    lr = conf['params']['lr']
    return optim.SGD(model.trainable_parameters(), lr = lr)

