import numpy as np
import torch.nn as nn
import random
import torch
import torch.optim as optim
from Seq2seq.models.utils import *
import tqdm
from Seq2seq.models.model import *
from Seq2seq.models.dataset import UIdataset
from Seq2seq.models.uiconfig import *


def training(model, optim,schedular, criterion, train_iter, epoch, vocab_size):

    model.train()
    losses = []
    label = []
    preds = []
    softmax = nn.Softmax(dim = -1)
    print('\nTrain_Epoch:', epoch)
    for batch in tqdm.tqdm(train_iter):
        optim.zero_grad()
        mask = create_masks(batch)
        input = batch.seq_x.cuda()
        truelabel_cls = batch.seq_y.transpose(0,1).cuda()
        attn_mask = mask.cuda()

        logits_cls = model(input, attn_mask)

        loss_cls = criterion(logits_cls.view(-1, vocab_size), truelabel_cls.contiguous().view(-1, ))

        loss = loss_cls

        losses.append(loss.item())

        #for now we are only interested in accuracy and f1 of the classification task
        preds_cls = softmax(logits_cls).argmax(2)   #to get a vector of probabilties for each word

        #append labels & preds of each batch to calculate score for every epoch:
        nptrue, nppreds = prune_preds(truelabel_cls.contiguous() .view(-1), preds_cls.view(-1))

        label.extend(nptrue)
        preds.extend(nppreds)

        loss.backward()

        optim.step()

        if schedular is not None:
            schedular.step()

    return losses, label, preds

def validation(model, criterion, valid_iter, epoch, vocab_size):
    model.eval()
    losses = []
    label = []
    preds = []

    softmax = nn.Softmax(dim=-1)
    print('\nValid_Epoch:', epoch)

    with torch.no_grad():
        for batch in tqdm.tqdm(valid_iter):


            mask = create_masks(batch)
            input = batch.seq_x.cuda()
            truelabel_cls = batch.seq_y.transpose(0,1).cuda()
            attn_mask = mask.cuda()
            logits_cls = model(input, attn_mask)
            loss_cls = criterion(logits_cls.view(-1, vocab_size), truelabel_cls.contiguous().view(-1, ))
            loss = loss_cls

            losses.append(loss.item())

            preds_cls = softmax(logits_cls).argmax(2)

            #plot_save(truelabel_cls, preds_cls, epoch)

            nptrue, nppreds = prune_preds(truelabel_cls.contiguous().view(-1), preds_cls.view(-1))

            label.extend(nptrue)
            preds.extend(nppreds)


    return losses, label, preds,

def train_val(train_iter, valid_iter, vocab_size, model_path:str, trial=None, best_params=None):

    epochs = UIconfig.epochs
    lrmain = UIconfig.lr_main
    drop_out = UIconfig.drop_out
    print({'lrmain':lrmain, 'drop_out':drop_out})

    model = Transformer(vocab_size, UIconfig.d_model, UIconfig.num_encoder_layers, UIconfig.nhead, drop_out)
    # this is how they initialize params in paper
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    optimizer = optim.Adam(model.parameters(), lr=lrmain)

    criterion = nn.CrossEntropyLoss(ignore_index= UIconfig.ignore_index)
    model.cuda()
    score = score_cal()

    for epoch in range(epochs):
        train_losses, label, preds = training(model, optimizer, None, criterion, train_iter, epoch, vocab_size)
        f1, acc = f1score(label, preds, 'weighted')
        score.train_f1.append(f1)
        score.train_acc.append(acc)
        score.train_loss.append(sum(train_losses)/len(train_losses))
        print('train_weighted_f1', f1)
        print('train_acc', acc)

        valid_loss, valid_label, valid_preds = validation(model, criterion, valid_iter, epoch, vocab_size)
        valid_f1, valid_acc = f1score(valid_label, valid_preds, 'weighted')
        score.valid_f1.append(valid_f1)
        score.valid_acc.append(valid_acc)
        score.valid_loss.append(sum(valid_loss) / len(valid_loss))

        print('valid_weighted_f1:', valid_f1)
        print('valid_acc:', valid_acc)
        #classificationreport(valid_label, valid_preds)

    if(trial is None):
        print('-saving model-')
        print(model_path)
        torch.save(model, model_path)


    return score



def main():
    print('hello')
    model_path = '../models/results/model.tar'

    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    np.random.seed(UIconfig.seed)
    torch.manual_seed(UIconfig.seed)
    random.seed(UIconfig.seed)


    dataset = UIdataset()

    train_iter, valid_iter, test_iter = dataset.get_loaders()
    score = train_val(train_iter, valid_iter, dataset.get_vocab_size(), model_path, None, None)
    print_result(score, UIconfig.epochs)



if __name__ == "__main__":
    main()