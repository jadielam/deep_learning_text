class Callback():
    def __init__(self):
        pass
    
    def on_iter_begin(iter_idx, epoch_idx, model, statistics):
        pass

    def on_epoch_begin(epoch_idx, model, statistics):
        pass

    def on_epoch_end(epoch_idx, model, statistics):
        pass

    def on_iter_end(iter_idx, epoch_idx, model, statistics):
        pass
    
    def on_train_end(model, statistics):
        pass

# TODO: Callbacks to implement:
#1. Print callback
class PrintCallback(Callback):
    def __init__(self):
        super(PrintCallback, self).__init__()
    
    def on_iter_end(iter_idx, epoch_idx, model, iter_stats):
        tr_loss = iter_stats.get_stat(iter_idx, "train_loss")
        total_iters = iter_stats.capacity
        epoch_nb = epoch_idx + 1
        iter_nb = iter_idx + 1
        print("Epoch # {} - {}/{} - training loss: {0:.4f}".format(epoch_nb, iter_nb, total_iters, tr_loss), end = '\r')

    def on_epoch_end(epoch_idx, model, epoch_stats):
        tr_loss = epoch_stats.get_stat(epoch_idx, "train_loss")
        val_loss = epoch_stats.get_stat(epoch_idx, "val_loss")
        total_epochs = epoch_stats.capacity
        epoch_nb = epoch_idx + 1
        print("Epoch # {} - training loss: {0:.4f} - validation loss: {0:.4f}".format(epoch_nb, tr_loss, val_loss))
        

#2. Statistics save callback

#3. Model save callback
class ModelSaveCallback(Callback):
    def __init__(self):
        super(ModelSaveCallback, self).__init__()
    
    def on_epoch_end(epoch_idx, model, epoch_stats):
        #TODO: 
        #1. Get validation stats and save model if last val is best one
        #2. If val stats have None values in them, then use trainig stats 
        #instead.
        return