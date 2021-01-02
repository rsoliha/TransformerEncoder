from sklearn.metrics import *
from matplotlib import pyplot
import pickle
import json
from Seq2seq.models.uiconfig import *


def save_params(best_params, name):
    with open(name, 'wb') as handle:
        pickle.dump(best_params, handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_json(name):
    with open(name, 'rb') as handle:
        best_params_dict = pickle.load(handle)
    return best_params_dict

def save_json(params, path):
    with open(path, 'w') as fp:
        json.dump(params, fp)

def load_param(path):
    with open(path, 'r') as fp:
        params = json.load(fp)
    return params

def f1score(y_true, y_pred, average = 'macro'):

    f1 = f1_score(y_true, y_pred, average=average)
    acc = accuracy_score(y_true, y_pred)
    return f1, acc

def classificationreport(y_true, y_pred):
    print(classification_report(y_true, y_pred))

class score_cal:
    def __init__(self):
        self.train_f1 = []
        self.valid_f1 = []
        self.train_acc = []
        self.valid_acc = []
        self.train_loss = []
        self.valid_loss = []

def print_result(score:score_cal, epochs):
    epoch_vals = [i + 1 for i in range(epochs)]
    pyplot.subplot(311)
    pyplot.title("Loss")
    pyplot.plot(epoch_vals, score.train_loss, label='train')
    pyplot.plot(epoch_vals, score.valid_loss, label='valid')
    pyplot.legend()
    pyplot.xticks(epoch_vals)

    if(score.train_f1 != []):
        pyplot.subplot(312)
        pyplot.title("F1")
        pyplot.plot(epoch_vals, score.train_f1, label='train')
        pyplot.plot(epoch_vals, score.valid_f1, label='valid')
        pyplot.legend()
        pyplot.xticks(epoch_vals)

        pyplot.subplot(313)
        pyplot.title("acc")
        pyplot.plot(epoch_vals, score.train_acc, label='train')
        pyplot.plot(epoch_vals, score.valid_acc, label='valid')
        pyplot.legend()
        pyplot.xticks(epoch_vals)

    pyplot.show()


def prune_preds(y_true, y_pred):
    true = []
    preds = []
    y_true = y_true.cpu().detach().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    for i in range(len(y_true)):
        if(y_true[i] != UIconfig.ignore_index): # ignore padding index
            true.append(y_true[i])
            preds.append(y_pred[i])
    return true, preds

def create_masks(batch):
    input_seq = batch.seq_x.transpose(0,1)
    input_pad = 1
    # creates mask with 1s wherever there is padding in the input
    input_msk = (input_seq == input_pad)
    return input_msk

def plot_save(ytrue, ypred):
    print(ytrue[0])
    print(ypred[0])