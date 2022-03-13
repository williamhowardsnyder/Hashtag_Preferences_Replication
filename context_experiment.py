import torch
import argparse
import logging
import json
import time
import numpy as np
import random
from Modules.mlp import Mlp
from itertools import chain
from torch.optim import Adam
import re
from tqdm import tqdm
from utils.scratch_dataset import my_collate
from utils.scratch_dataset import ScratchDataset
import torch.utils.data as data
import torch.nn.functional as F
import argparse
from train import train_mlp
from pykp.model import Seq2SeqModel, NTM
from process_opt import process_opt
from utils.time_log import time_since
from data_loader import load_data_and_vocab
import os
from utils.config import my_own_opts
from utils.config import vocab_opts
from utils.config import model_opts
from utils.config import train_opts
from utils.config import init_logging
import nltk

import numpy as np
import matplotlib.pyplot as plt
nltk.download('punkt')

CUDA_LAUNCH_BLOCKING=1

torch.manual_seed(2021)
if torch.cuda.is_available():
    torch.cuda.manual_seed(2021)


parser = argparse.ArgumentParser(description='vae_train.py',
                                    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
my_own_opts(parser)
vocab_opts(parser)
model_opts(parser)
train_opts(parser)

opt = parser.parse_args()
opt = process_opt(opt)
opt.input_feeding = False
opt.copy_input_feeding = False


opt.device = torch.device("cuda:%d" % opt.gpuid)


def init_optimizers(model, ntm_model, opt):
    optimizer_seq2seq = Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=opt.learning_rate)
    optimizer_ntm = Adam(params=filter(lambda p: p.requires_grad, ntm_model.parameters()), lr=opt.learning_rate)
    whole_params = chain(model.parameters(), ntm_model.parameters())
    optimizer_whole = Adam(params=filter(lambda p: p.requires_grad, whole_params), lr=opt.learning_rate)

    return optimizer_seq2seq, optimizer_ntm, optimizer_whole

def invert_dict(d):
    return dict((v,k) for k,v in d.items())

def get_num_hshtg_context():
    """
    Returns a dictionary of hashtags to their counts in the training data
    """
    train_file = './Data/train.csv'
    tags2counts = {}
    f = open(train_file)
    for line in f:
        l = line.strip('\n').split('\t')
        text, user, hashtags = l[0], l[1], l[2:]

        for hashtag in hashtags:
            if len(hashtag) == 0:
                continue
            if hashtag not in tags2counts:
                tags2counts[hashtag] = 0
            tags2counts[hashtag] += 1

    counts2tags = {}
    for tag, count in tags2counts.items():
        if count not in counts2tags:
            counts2tags[count] = set()
        counts2tags[count].add(tag)

    # Uncomment this code to print out the frequency of hashtags per count
    # count_list = []
    # for i in range(50):
    #     if i in counts2tags.keys():
    #         val = len(counts2tags[i])
    #     else:
    #         val = 0
        
    #     print(i, "\t", val)

    
    return tags2counts, counts2tags


def plot_hashtag_prec():
    # Get the counts of each hashtag
    tags2counts, counts2tags = get_num_hshtg_context()
    idx2tags = {}
    for i in range(0, 6):
        tag_set = set()
        for j in range(i * 5 + 1, (i + 1) * 5 + 1):
            tag_set = tag_set.union(counts2tags[j])

        idx2tags[i] = tag_set


    output_path = "./Records/model_output.txt"
    f = open(output_path, "r")
    tag2scores = {}
    for line in f:
        arr = line.split("'")
        user = arr[1]
        tag = arr[3]
        arr = line.split("\t")
        pred = float(arr[1])
        true = float(arr[2].strip("\n").strip("[").strip("]"))

        # print(f"user: {user}\ttag: {tag}\tpred: {pred}\ttrue: {true}")

        if tag not in tag2scores:
            tag2scores[tag] = ([], [])
        tag2scores[tag][0].append(pred)
        tag2scores[tag][1].append(true)


    prec_list = []
    for i in range(0, 6):
        print("Getting precision for context between", i * 5 + 1, " and ", (i + 1) * 5, "...")
        tag_set = idx2tags[i]
        prec = 0.0
        total = 0
        for tag in tag_set:
            if tag not in tag2scores:
                continue
            pred, true = tag2scores[tag]
            # print(true)
            # print(pred)
            prec += precision_at_k(np.array(true), np.array(pred), 5)
            total += 1

        prec = prec / total
        print(f"\tprec averaged over {total} tags is {prec}")
        prec_list.append(prec)

    data = [[0.31, 0.32, 0.3, 0.325, 0.325, 0.475], # LSTM Values
            prec_list]
    X = np.arange(6)
    fig = plt.figure()
    ax = fig.add_axes([0.2,0.2,0.7,0.7])
    ax.bar(X + 0.00, data[1], color = 'blue', width = 0.35)
    ax.bar(X + 0.35, data[0], color = 'orange', width = 0.35)

    ax.set_ylabel('P@5')
    ax.set_xlabel('Number of Hashtag Context Tweets')
    ax.set_title('P@5 for Varied Hashtag Contexts')
    ax.set_xticks(X, ('0-5', '6-10', '11-15', '16-20', '21-25', '26-30'))
    ax.set_yticks(np.arange(0, 0.7, 0.1))
    ax.legend(labels=['Our Model', 'LSTM-ATT'])
    plt.savefig("hashtag_bar_plot.png")


def get_num_user_context():
    """
    Returns a dictionary of users to the number of tweets they've made in the training data
    """
    train_file = './Data/train.csv'
    users2counts = {}
    f = open(train_file)
    for line in f:
        l = line.strip('\n').split('\t')
        text, user, hashtags = l[0], l[1], l[2:]
        
        if user not in users2counts:
            users2counts[user] = 0
        users2counts[user] += 1

    counts2users = {}
    for user, count in users2counts.items():
        if count not in counts2users:
            counts2users[count] = set()
        counts2users[count].add(user)

    return users2counts, counts2users


def plot_user_prec():
    # Get the counts of each hashtag
    users2counts, counts2users = get_num_user_context()
    idx2users = {}
    for i in range(0, 5):
        user_set = set()
        for j in range(i * 7 + 1, (i + 1) * 7 + 1):
            user_set = user_set.union(counts2users[j])

        idx2users[i] = user_set
        # print(user_set)

    output_path = "./Records/model_output.txt"
    f = open(output_path, "r")
    user2scores = {}
    for line in f:
        arr = line.split("'")
        user = arr[1]
        tag = arr[3]
        arr = line.split("\t")
        pred = float(arr[1])
        true = float(arr[2].strip("\n").strip("[").strip("]"))

        # print(f"user: {user}\ttag: {tag}\tpred: {pred}\ttrue: {true}")

        if user not in user2scores:
            user2scores[user] = ([], [])
        user2scores[user][0].append(pred)
        user2scores[user][1].append(true)

    prec_list = []
    for i in range(0, 5):
        print("Getting precision for context between", i * 7 + 1, " and ", (i + 1) * 7, "...")
        user_set = idx2users[i]
        prec = 0.0
        total = 0
        for user in user_set:
            print(user)
            if user not in user2scores:
                continue
            pred, true = user2scores[user]
            # print(true)
            # print(pred)
            prec += precision_at_k(np.array(true), np.array(pred), 5)
            total += 1

        prec = prec / total
        print(f"\tprec averaged over {total} users is {prec}")
        prec_list.append(prec)

    data = [[0.105, 0.1, 0.09, 0.078, 0.076], # LSTM Values
            prec_list]
    X = np.arange(5)
    fig = plt.figure()
    ax = fig.add_axes([0.2,0.2,0.7,0.7])
    ax.bar(X + 0.00, data[1], color = 'blue', width = 0.35)
    ax.bar(X + 0.35, data[0], color = 'orange', width = 0.35)

    ax.set_ylabel('P@5')
    ax.set_xlabel('Number of User Context Tweets')
    ax.set_title('P@5 for Varied User Contexts')
    ax.set_xticks(X, ('1-7', '8-14', '15-21', '22-28', '29-36'))
    ax.set_yticks(np.arange(0, 0.18, 0.02))
    ax.legend(labels=['Our Model', 'LSTM-ATT'])
    plt.savefig("user_bar_plot.png")




def precision_at_k(y_true_arr, y_score_arr, k, pos_label=1):
    # Makes this compatible with various array types    
    y_true_arr = y_true_arr == pos_label
    
    desc_sort_order = np.argsort(y_score_arr)[::-1]
    y_true_sorted = y_true_arr[desc_sort_order]
    y_score_sorted = y_score_arr[desc_sort_order]
    
    true_positives = y_true_sorted[:k].sum()
    
    return true_positives / k



def main(opt):
    # read files
    with open('./Data/embeddings.json', 'r') as f:
        text_emb_dict = json.load(f)

    with open('./Data/userList.txt', "r") as f:
        x = f.readlines()[0]
        user_list = re.findall(r"['\'](.*?)['\']", str(x))

    with open(f'./NTMData/trainVaeEmbeddings.json', 'r') as f:
                train_vae_emb_dict = json.load(f)

    with open(f'./NTMData/validVaeEmbeddings.json', 'r') as f:
        valid_vae_emb_dict = json.load(f)

    test_vae_emb_dict = 0

    train_file = './Data/train.csv'
    valid_file = './Data/valid.csv'
    test_file = './Data/test.csv'

    start_time = time.time()
    train_data_loader, train_bow_loader, valid_data_loader, valid_bow_loader, \
    word2idx, idx2word, vocab, bow_dictionary = load_data_and_vocab(opt, load_train=True)

    vocab_dict = bow_dictionary

    opt.bow_vocab_size = len(bow_dictionary)

    model = Seq2SeqModel(opt).cuda()
    ntm_model = NTM(opt).cuda()

    best_epoch = 99

    #test the model
    print("test the model!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    test_dataset = ScratchDataset(data_split='Test', user_list=user_list, train_file=train_file, valid_file=valid_file, test_file=test_file, bert_dict=text_emb_dict, train_vae_dict=train_vae_emb_dict, valid_vae_dict=valid_vae_emb_dict, test_vae_dict=test_vae_emb_dict, bow_dictionary=invert_dict(bow_dictionary), joint_train_flag=True)
    

    # Get the counts of each hashtag
    tags2counts, counts2tags = get_num_hshtg_context()
    idx2tags = {}
    for i in range(0, 6):
        tag_set = set()
        for j in range(i * 5 + 1, (i + 1) * 5 + 1):
            tag_set = tag_set.union(counts2tags[j])

        idx2tags[i] = tag_set

    best_model = Mlp(ntm_model, vocab_dict, 768, 256)
    best_model.load_state_dict(torch.load('./ModelRecords'+f'/model_{best_epoch}.pt'))
    if torch.cuda.is_available():
        best_model = best_model.cuda()
    
    best_model.eval()
    fr = open('./Records/test.dat', 'r')
    fws = []
    for i in idx2tags.keys():
        f = open(f'./Records/test2_{i * 5 + 1}_{(i + 1) * 5}.dat', 'w')
        fws.append(f)

    preFs = []
    for i in idx2tags.keys():
        preF = open(f'./Records/pre_{i * 5 + 1}_{(i + 1) * 5}.txt', "w")
        preFs.append(preF)

    lines = fr.readlines()
    lines = [line.strip() for line in lines if line[0] != '#']
    last_user = lines[0][6:]
    last_user = last_user.split(' ')[0]

    # Init all the files with the first user
    for fw in fws:
        print('# query 0', file=fw)
    
    with torch.no_grad():
        for i in tqdm(range(len(test_dataset))):
            
            test_user_feature, test_hashtag_feature, test_bow_hashtag_feature, test_label = test_dataset[i]
            test_user_feature = test_user_feature.cuda()
            test_hashtag_feature = test_hashtag_feature.cuda()
            test_bow_hashtag_feature = test_bow_hashtag_feature.cuda()
            test_label = test_label.cuda()

            tag = test_dataset.user_hashtag[i][1]
            if tag not in tags2counts:
                continue
            count = tags2counts[tag]
            if count > 30:
                continue

            line = lines[i]
            user = line[6:]
            user = user.split(' ')[0]
            if (user == last_user):
                pass
            else:
                for fw in fws:
                    print('# query ' + user, file=fw)
                    last_user = user

            fw = fws[int((count-1)/5)]
            preF = preFs[int((count-1)/5)]

            weight, topic_words, pred_label, recon_batch, data_bow, mu, logvar = \
                    best_model(True, test_user_feature.unsqueeze(0), torch.tensor([len(test_user_feature)]), \
                               test_hashtag_feature.unsqueeze(0), torch.tensor([len(test_hashtag_feature)]), \
                               test_bow_hashtag_feature.unsqueeze(0))

            print(line, file=fw)
            pred_label = pred_label.cpu().detach().numpy().tolist()[0]
            preF.write(f"{pred_label}\n")
        
    for preF in preFs:
        preF.close()
    for fw in fws:
        fw.close()
            
    return


# main(opt)
plot_hashtag_prec()
plot_user_prec()