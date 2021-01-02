from Seq2seq.models.dataset import UIdataset
import torch
import torch.nn as nn
import tqdm
from Seq2seq.models.utils import *
import random
import matplotlib.pyplot as plt
import matplotlib

def get_orignal_data(dataset, inp):
    l = []
    for i in inp:
        l.append(dataset.get_vocab_item(i))
    return l

def plot(i_list, p_list, t_list, dataset):
    for i in range(len(i_list)):
        inp = i_list[i]
        p = p_list[i]
        t = t_list[i]
        if(1 in inp):
            index_1 = inp.index(1)
            inp = inp[:index_1]
            p = p[:index_1]
            t = t[:index_1]
        inp_dict=get_orignal_data(dataset, inp)
        pred_dict=get_orignal_data(dataset, p)
        true_dict=get_orignal_data(dataset, t)
        print(inp_dict)
        print(pred_dict)

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        fig.suptitle('True vs Input vs Predictions')

        for j in range(0,len(inp),5):
            orig_elt=true_dict[j+0]
            orig_x=int(true_dict[j+1][2:])
            orig_y=int(true_dict[j+2][2:])
            orig_h=int(true_dict[j+3][2:])
            orig_w=int(true_dict[j+4][2:])

            inp_elt = inp_dict[j + 0]
            inp_x = int(inp_dict[j + 1][2:])
            inp_y = int(inp_dict[j + 2][2:])
            inp_h = int(inp_dict[j + 3][2:])
            inp_w = int(inp_dict[j + 4][2:])

            pred_elt = pred_dict[j + 0]
            pred_x = int(pred_dict[j + 1][2:])
            pred_y = int(pred_dict[j + 2][2:])
            pred_h = int(pred_dict[j + 3][2:])
            pred_w = int(pred_dict[j + 4][2:])

            rect1 = matplotlib.patches.Rectangle((orig_x, orig_y),
                                                 orig_w, orig_h,
                                                 color='black', fill=0)
            ax1.add_patch(rect1)

            rect2 = matplotlib.patches.Rectangle((inp_x, inp_y),
                                                 inp_w, inp_h,
                                                 color='black', fill=0)
            ax2.add_patch(rect2)

            rect3 = matplotlib.patches.Rectangle((pred_x, pred_y),
                                                 pred_w, pred_h,
                                                 color='black', fill=0)
            ax3.add_patch(rect3)
        ax1.set_xlim([-1, 20])
        ax1.set_ylim([-1, 20])
        ax2.set_xlim([-1, 20])
        ax2.set_ylim([-1, 20])
        ax3.set_xlim([-1, 20])
        ax3.set_ylim([-1, 20])
        plt.show()









def testing(test_iter, model, vocab_size):
    model.eval()
    preds = []
    label = []
    softmax = nn.Softmax(dim=-1)
    i_list = []
    p_list = []
    t_list = []
    with torch.no_grad():
        for batch in tqdm.tqdm(test_iter):
            # generate random num between 0 and bs(32)
            #randnum = random.randint(0, 31)
            input_ids = batch.seq_x.cuda()
            truelabel_cls = batch.seq_y.transpose(0,1).cuda()
            mask = create_masks(batch)
            att_mask = mask.cuda()
            logits_cls = model(input_ids, att_mask)
            prediction = softmax(logits_cls).argmax(2)
            inp = input_ids.transpose(0,1)
            print(inp.size())
            if(inp.size()[0]==32):          #batch size
                 #i = inp[randnum].cpu().detach().numpy().tolist()
                 #p = prediction[randnum].cpu().detach().numpy().tolist()
                 #t = truelabel_cls[randnum].cpu().detach().numpy().tolist()
                 for i in range(32):
                     input = inp[i].cpu().detach().numpy().tolist()
                     p = prediction[i].cpu().detach().numpy().tolist()
                     t = truelabel_cls[i].cpu().detach().numpy().tolist()
                     if (1 in input):
                         index_1 = input.index(1)
                         if(index_1 < 50):      #plot those with less than 10 elements
                             i_list.append(input)
                             p_list.append(p)
                             t_list.append(t)

            nptrue, nppreds = prune_preds(truelabel_cls.contiguous().view(-1), prediction.view(-1))
            label.extend(nptrue)
            preds.extend(nppreds)

    return label, preds, i_list, p_list, t_list






def main():
    device = torch.device("cuda:1")
    torch.cuda.set_device(device)
    model_path = '../models/results/model.tar'
    print(model_path)
    dataset = UIdataset()
    vocab_size = dataset.get_vocab_size()
    _, _, test_iter = dataset.get_loaders()
    transformer = torch.load(model_path)
    print(len(test_iter))
    label, preds,i , p, t = testing(test_iter, transformer, vocab_size)
    f1, acc = f1score(label, preds, average='weighted')

    print('test_acc:', acc)
    print('test_f1:', f1)

    plot(i, p, t, dataset)


if __name__ == "__main__":
    main()