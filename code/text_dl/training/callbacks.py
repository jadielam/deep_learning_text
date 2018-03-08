import numpy as np
import torch
import time

class Callback():
    def __init__(self):
        pass
    
    def on_iter_begin(self, iter_idx, epoch_idx, model, iter_stats):
        pass

    def on_epoch_begin(self, epoch_idx, model, epoch_stats):
        pass

    def on_epoch_end(self, epoch_idx, model, epoch_stats):
        pass

    def on_iter_end(self, iter_idx, epoch_idx, model, iter_stats):
        pass
    
    def on_train_end(self, model, train_stats):
        pass

#1. Print callback
class PrintCallback(Callback):
    def __init__(self):
        super(PrintCallback, self).__init__()

    def on_epoch_begin(self, epoch_idx, model, epoch_stats):
        self.epoch_start_time = time.time()
    
    def on_iter_begin(self, iter_idx, epoch_idx, model, iter_stats):
        self.iter_start_time = time.time()

    def on_iter_end(self, iter_idx, epoch_idx, model, iter_stats):
        iter_end_time = time.time()
        iter_elapsed_time = iter_end_time = self.iter_start_time

        tr_loss = iter_stats.get_stat(iter_idx, "train_loss")
        total_iters = iter_stats.capacity
        epoch_nb = epoch_idx + 1
        iter_nb = iter_idx + 1
        current_time = time.time()
        
        epoch_estimated_time_remaining = (total_iters - iter_nb) * iter_elapsed_time
        print("Epoch # {} - {}/{} - time remaining: {:.2f} - training loss: {:.4f}".format(epoch_nb, iter_nb, total_iters, epoch_estimated_time_remaining, tr_loss), end = "\r")

    def on_epoch_end(self, epoch_idx, model, epoch_stats):
        epoch_end_time = time.time()
        epoch_elapsed_time = epoch_end_time - self.epoch_start_time

        tr_loss = epoch_stats.get_stat(epoch_idx, "train_loss")
        val_loss = epoch_stats.get_stat(epoch_idx, "val_loss")
        if val_loss is None:
            val_loss = -1.
        total_epochs = epoch_stats.capacity
        epoch_nb = epoch_idx + 1
        print("Epoch # {} - seconds: {:.2f} - training loss: {:.4f} - validation loss: {:.4f}".format(epoch_elapsed_time, epoch_nb, tr_loss, val_loss))    


class HistorySaveCallback(Callback):
    def __init__(self, output_path = "history.csv", 
                statistics = ["train_loss", "val_loss"]):
        super(HistorySaveCallback, self).__init__()
        self.output_path = output_path
        self.statistics = statistics
    
    def __hash__(self):
        return 0
    
    def __eq__(self, other):
        return isinstance(other, HistorySaveCallback)
    
    def on_epoch_end(self, epoch_idx, model, epoch_stats):
        if epoch_idx == 0:
            with open(self.output_path, "a+") as f:
                f.write(",".join(["epoch"] + self.statistics))
                f.write('\n')
        
        entries = []
        for stat_name in self.statistics:
            try:
                val = epoch_stats.get_stat(epoch_idx, stat_name)
            except KeyError:
                val = None
            entries.append(val)
        with open(self.output_path, "a") as f:
            f.write(",".join([str(epoch_idx + 1)] + [str(a) for a in entries]))
            f.write("\n")

#3. Model save callback
class ModelSaveCallback(Callback):
    def __init__(self, model_output_path = "model.pth"):
        super(ModelSaveCallback, self).__init__()
        self.model_output_path = model_output_path
    
    def __hash__(self):
        return 0

    def __eq__(self, other):
        return isinstance(other, ModelSaveCallback)
    
    def on_epoch_end(self, epoch_idx, model, epoch_stats):
        stat_to_use = 'val_loss'
        try:
            current_loss = epoch_stats.get_stat(epoch_idx, stat_to_use)
            if current_loss is None:
                stat_to_use = 'train_loss'
        except KeyError:
            stat_to_use = "train_loss"
        
        try:
            best_loss_idx = np.argmin([epoch_stats.get_stat(i, stat_to_use) for i in range(epoch_idx + 1)])
            if best_loss_idx == epoch_idx:
                torch.save(model.state_dict(), self.model_output_path)
        except KeyError:
            pass
        