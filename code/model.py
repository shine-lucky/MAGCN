import torch
from torch import nn
from torch_geometric.nn import GCNConv
torch.backends.cudnn.enabled = False
from utils1 import *


class MAGCN(nn.Module):
    def __init__(self, args):
        """
        :param args: Arguments object.
        """
        super(MAGCN, self).__init__()
        self.args = args

        self.gcn_11 = GCNConv(self.args.f0, self.args.out_channels)
        self.gcn_12 = GCNConv(self.args.out_channels, self.args.out_channels)
       


        self.gcn_1 = GCNConv(self.args.f0, self.args.out_channels)
        self.gcn_2 = GCNConv(self.args.out_channels, self.args.out_channels)
       

        self.globalAvgPool_y = nn.AvgPool2d((self.args.out_channels, self.args.mirna_size), (1, 1))
        self.globalAvgPool_z = nn.AvgPool2d((self.args.out_channels, self.args.disease_size), (1, 1))
        self.globalAvgPool_x = nn.AvgPool2d((self.args.out_channels, self.args.lncrna_size), (1, 1))
        self.fc1_y = nn.Linear(in_features=self.args.view * self.args.gcn_layers,
                               out_features=6 *self.args.view * self.args.gcn_layers)
        self.fc2_y = nn.Linear(in_features=6 * self.args.view * self.args.gcn_layers,
                               out_features=self.args.view * self.args.gcn_layers)

        self.fc1_x = nn.Linear(in_features=self.args.gcn_layers,
                               out_features=6 * self.args.gcn_layers)
        self.fc2_x = nn.Linear(in_features=6 * self.args.gcn_layers,
                               out_features=self.args.gcn_layers)

        self.fc1_z = nn.Linear(in_features=self.args.gcn_layers,
                               out_features=6 * self.args.gcn_layers)
        self.fc2_z = nn.Linear(in_features=6 * self.args.gcn_layers,
                               out_features=self.args.gcn_layers)

        self.sigmoidx = nn.Sigmoid()
        self.sigmoidy = nn.Sigmoid()
        self.sigmoidz = nn.Sigmoid()

        self.cnn_y = nn.Conv1d(in_channels=self.args.view * self.args.gcn_layers,
                               out_channels=self.args.out_channels,
                               kernel_size=(self.args.out_channels, 1),
                               stride=1,
                               bias=True)
        self.cnn_x = nn.Conv1d(in_channels=self.args.gcn_layers,
                               out_channels=self.args.out_channels,
                               kernel_size=(self.args.out_channels, 1),
                               stride=1,
                               bias=True)
        self.cnn_z = nn.Conv1d(in_channels=self.args.gcn_layers,
                               out_channels=self.args.out_channels,
                               kernel_size=(self.args.out_channels, 1),
                               stride=1,
                               bias=True)



        self.decoder1 = nn.Linear(in_features=self.args.out_channels,
                               out_features=self.args.out_channels)
        self.decoder2 = nn.Linear(in_features=self.args.out_channels,
                                  out_features=self.args.out_channels)

    
    def forward(self, data):
        torch.manual_seed(1)
        adj_m_d = data['Adj_m_d']
        adj_l_m = data['Adj_l_m']
        m = torch.randn(self.args.mirna_size, self.args.f0)
        d = torch.randn(self.args.disease_size, self.args.f0)
        l = torch.randn(self.args.lncrna_size, self.args.f0)

        m_d = torch.cat((m, d), 0)
        l_m = torch.cat((l, m), 0)

        H11 = torch.relu(self.gcn_11(l_m.cuda(), adj_l_m['edge_index'].cuda(),
                                   adj_l_m['data'][adj_l_m['edge_index'][0], adj_l_m['edge_index'][1]].cuda()))
        H12 = torch.relu(self.gcn_12(H11.cuda(), adj_l_m['edge_index'].cuda(),
                                   adj_l_m['data'][adj_l_m['edge_index'][0], adj_l_m['edge_index'][1]].cuda()))



        H1 = torch.relu(self.gcn_1(m_d.cuda(), adj_m_d['edge_index'].cuda(),
                                   adj_m_d['data'][adj_m_d['edge_index'][0], adj_m_d['edge_index'][1]].cuda()))
        H2 = torch.relu(self.gcn_2(H1.cuda(), adj_m_d['edge_index'].cuda(),
                                   adj_m_d['data'][adj_m_d['edge_index'][0], adj_m_d['edge_index'][1]].cuda()))




        YM = torch.cat((H1[:self.args.mirna_size], H2[:self.args.mirna_size],H11[self.args.lncrna_size:],H12[self.args.lncrna_size:]), 1).t()
        YM = YM.view(1, self.args.view * self.args.gcn_layers, self.args.out_channels, -1)
        y_channel_attenttion = self.globalAvgPool_y(YM)
        y_channel_attenttion = y_channel_attenttion.view(y_channel_attenttion.size(0), -1)
        y_channel_attenttion = self.fc1_y(y_channel_attenttion)
        y_channel_attenttion = torch.relu(y_channel_attenttion)
        y_channel_attenttion = self.fc2_y(y_channel_attenttion)
        y_channel_attenttion = self.sigmoidy(y_channel_attenttion)
        y_channel_attenttion = y_channel_attenttion.view(y_channel_attenttion.size(0), y_channel_attenttion.size(1), 1,
                                                         1)
        YM_channel_attention = y_channel_attenttion * YM
        YM_channel_attention = torch.relu(YM_channel_attention)


        ZD = torch.cat((H1[self.args.mirna_size:], H2[self.args.mirna_size:]), 1).t()


        ZD = ZD.view(1, self.args.gcn_layers, self.args.out_channels, -1)
        z_channel_attenttion = self.globalAvgPool_z(ZD)
        z_channel_attenttion = z_channel_attenttion.view(z_channel_attenttion.size(0), -1)
        z_channel_attenttion = self.fc1_z(z_channel_attenttion)
        z_channel_attenttion = torch.relu(z_channel_attenttion)
        z_channel_attenttion = self.fc2_z(z_channel_attenttion)
        z_channel_attenttion = self.sigmoidz(z_channel_attenttion)
        z_channel_attenttion = z_channel_attenttion.view(z_channel_attenttion.size(0), z_channel_attenttion.size(1), 1,
                                                         1)
        ZD_channel_attention = z_channel_attenttion * ZD
        ZD_channel_attention = torch.relu(ZD_channel_attention)



        XL = torch.cat((H11[:self.args.lncrna_size], H12[:self.args.lncrna_size:]), 1).t()
        XL = XL.view(1, self.args.gcn_layers, self.args.out_channels, -1)
        x_channel_attenttion = self.globalAvgPool_x(XL)
        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), -1)
        x_channel_attenttion = self.fc1_x(x_channel_attenttion)
        x_channel_attenttion = torch.relu(x_channel_attenttion)
        x_channel_attenttion = self.fc2_x(x_channel_attenttion)
        x_channel_attenttion = self.sigmoidx(x_channel_attenttion)
        x_channel_attenttion = x_channel_attenttion.view(x_channel_attenttion.size(0), x_channel_attenttion.size(1), 1, 1)
        XL_channel_attention = x_channel_attenttion * XL

        XL_channel_attention = torch.relu(XL_channel_attention)





        y = self.cnn_y(YM_channel_attention)
        y = y.view(self.args.out_channels, self.args.mirna_size).t()


        z = self.cnn_z(ZD_channel_attention)
        z = z.view(self.args.out_channels, self.args.disease_size).t()


        x = self.cnn_x(XL_channel_attention)
        x = x.view(self.args.out_channels, self.args.lncrna_size).t()


        out = self.decoder1(y)
        out = out.mm(z.t())
        out = torch.sigmoid(out)

        out1 = self.decoder2(x)
        out1 = out1.mm(y.t())
        out1 = torch.sigmoid(out1)



        return out, out1
