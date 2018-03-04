import numpy as np

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

#1. Print callback
class PrintCallback(Callback):
    def __init__(self):
        super(PrintCallback, self).__init__()
    
    def on_iter_end(iter_idx, epoch_idx, model, iter_stats):
        tr_loss = iter_stats.get_stat(iter_idx, "train_loss")
        total_iters = iter_stats.capacity
        epoch_nb = epoch_idx + 1
        iter_nb = iter_idx + 1
        print("Epoch # {} - {}/{} - training loss: {0:.4f}".format(epoch_nb, iter_nb, total_iters, tr_loss), end = "\r")

    def on_epoch_end(epoch_idx, model, epoch_stats):
        tr_loss = epoch_stats.get_stat(epoch_idx, "train_loss")
        val_loss = epoch_stats.get_stat(epoch_idx, "val_loss")
        total_epochs = epoch_stats.capacity
        epoch_nb = epoch_idx + 1
        print("Epoch # {} - training loss: {0:.4f} - validation loss: {0:.4f}".format(epoch_nb, tr_loss, val_loss))

class HistorySaveCallback(Callback):
    def __init__(self, output_path = "history.csv", 
                statistics = ["train_loss", "val_loss"]):
        super(StatisticsSaveCallback, self).__init__()
        self.output_path = output_path
        self.statistics = statistics
    
    def on_epoch_end(epoch_idx, model, epoch_stats):
        if epoch_idx == 0:
            with open(self.output_path, "a+") as f:
                f.write(",".join(["epoch"] + statistics))
                f.write('\n')
        
        entries = []
        for stat_name in self.statistics:
            try:
                val = epoch_stats.get_stat(epoch_idx, stat_name)
            except KeyError:
                val = None
            entries.append(val)
        with open(self.output_path, "a"):
            f.write(",".join([epoch_idx + 1] + entries))
            f.write("\n")

#3. Model save callback
class ModelSaveCallback(Callback):
    def __init__(self, model_output_path = "model.pth"):
        super(ModelSaveCallback, self).__init__()
        self.model_output_path = model_output_path
    
    def on_epoch_end(epoch_idx, model, epoch_stats):
        stat_to_use = 'val_loss'
        try:
            current_loss = epoch_stats.get_stat(epoch_idx, stat_to_use)
            if current_loss is None:
                stat_to_use = 'train_loss'
        except KeyError:
            stat_to_use = "train_loss"
        
        try:
            best_loss_idx = np.argmin([epoch_stats.get_stat(i, stat_to_use) for i in range(epoch_idx)])
            if best_loss_idx == epoch_idx:
                torch.save(model.stat_dict(), self.model_output_path)
        except KeyError:
            pass
        