import numpy as np

import gc

from utils import load_miRNA_disease, load_incRNA_microRNA, constructNet, get_edge_index
import torch as t
from torch import optim
from loss import Myloss
from param import parameter_parser
from model import MAGCN


def train(model, train_data, optimizer, args):
    scheduler = t.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.8)
    model.train()
    regression_crit = Myloss()

    def train_epoch():
        model.zero_grad()
        score, score1 = model(train_data)

        loss1 = t.nn.MSELoss(reduction='mean')
        loss1 = loss1(score1, train_data['feature'].cuda())

        loss2 = regression_crit(train_data['Y_train'].cuda(), score)
        loss = loss2 + args.alpha * loss1
        loss = loss.requires_grad_()
        loss.backward(retain_graph=True)
        optimizer.step()
        scheduler.step()
        return loss

    for epoch in range(1, args.epoch + 1):
        train_reg_loss = train_epoch()
        scheduler.step()
        print("epoch : %d, loss:%.2f" % (epoch, train_reg_loss.item()))
    pass


def PredictScore(train_matrix, l_m, args):
    np.random.seed(args.seed)
    train_data = {}
    train_data['Y_train'] = t.Tensor(train_matrix)

    adj_m_d = constructNet(train_matrix)
    adj_m_d = t.Tensor(adj_m_d)
    adj_m_d_edge_index = get_edge_index(adj_m_d)
    train_data['Adj_m_d'] = {'data': adj_m_d, 'edge_index': adj_m_d_edge_index}

    adj_l_m = constructNet(l_m)
    adj_l_m = t.Tensor(adj_l_m)
    adj_l_m_edge_index = get_edge_index(adj_l_m)
    train_data['Adj_l_m'] = {'data': adj_l_m, 'edge_index': adj_l_m_edge_index}

    X = t.Tensor(l_m)
    train_data['feature'] = X

    model = MAGCN(args)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=args.learn_rate, weight_decay=args.weight_decay)

    train(model, train_data, optimizer, args)
    return model(train_data)


if __name__ == "__main__":
    m_d = load_miRNA_disease()
    l_m = load_incRNA_microRNA()
    args = parameter_parser()
