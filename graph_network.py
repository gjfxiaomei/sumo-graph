import dgl
import torch
from torch.cuda import init
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, edge_type):
        super(GATLayer, self).__init__()
        self.g = g
        # equation (1)
        self.edge_type = edge_type
        self.fc = nn.Linear(in_dim, out_dim, bias=False)
        # equation (2)
        self.attn_fc = nn.Linear(2*out_dim, 1, bias=False)

    def edge_attention(self, edges):
        # edge UDF for equation (2)
        z2 = torch.cat([edges.src['z'], edges.dst['z']], dim=-1)
        a = self.attn_fc(z2)
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        # message UDF for equation (3) & (4)
        return {'z': edges.src['z'], 'e': edges.data['e']}

    def reduce_func(self, nodes):
        # reduce UDF for equation (3) & (4)
        # equation (3)
        alpha = F.softmax(nodes.mailbox['e'], dim=1)
        # equation (4)
        h = torch.sum(alpha * nodes.mailbox['z'], dim=1)
        return {'h': h}

    def forward(self, h):
        # equation (1)
        z = self.fc(h)
        self.g.ndata['z'] = z
        # equation (2)
        self.g.apply_edges(self.edge_attention, etype=self.edge_type)
        # equation (3) & (4)
        self.g.update_all(
                self.message_func, self.reduce_func, etype=self.edge_type)
        return self.g.ndata.pop('h')


class MultiHeadGATLayer(nn.Module):
    def __init__(self, g, in_dim, out_dim, num_heads, edge_type, merge='cat'):
        super(MultiHeadGATLayer, self).__init__()
        self.heads = nn.ModuleList()
        for i in range(num_heads):
            self.heads.append(GATLayer(g, in_dim, out_dim, edge_type))
        self.merge = merge

    def forward(self, h):
        head_outs = [attn_head(h) for attn_head in self.heads]
        if self.merge == 'cat':
            # concat on the output feature dimension (dim=-1)
            return torch.cat(head_outs, dim=-1)
        else:
            # merge using average
            return torch.mean(torch.stack(head_outs))

class GraphNetwork(nn.Module):
    def __init__(self, g, in_dim, hidden_dim, out_dim, num_heads, edge_type):
        super(GraphNetwork, self).__init__()
        self.g = g
        self.layer1 = MultiHeadGATLayer(g, in_dim, hidden_dim, num_heads, edge_type)
        # Be aware that the input dimension is hidden_dim*num_heads since
        # multiple head outputs are concatenated together. Also, only
        # one attention head in the output layer.
        self.layer2 = MultiHeadGATLayer(g, hidden_dim * num_heads, out_dim, 1, edge_type)

    def forward(self, h):
        h = self.layer1(h)
        h = F.elu(h)
        h = self.layer2(h)
        return h


E_IN  = 0
N_IN  = 1
S_IN  = 2
W_IN  = 3

E_OUT = 4
N_OUT = 5
S_OUT = 6
W_OUT = 7

class GraphAgent(object):
    def __init__(self, phase_list, in_dim, hidden_dim, out_dim, num_heads, lr, edge_type):
        self.edge_type = edge_type
        self.agent_dict = {}
        self.optimizer_dict = {}
        for phase in phase_list:
            g = self.generate_graph(phase)
            net = GraphNetwork(g, in_dim, hidden_dim, out_dim, num_heads, edge_type)
            self.agent_dict[phase] = net
            self.optimizer_dict[phase] = torch.optim.Adam(net.parameters(), lr=lr)
    
    def train(self, features, labels, phase):
        if len(features) == 0 or len(labels) == 0 or len(features) != len(labels):
            return
        features = torch.tensor(features, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.float32)
        predicts = self.agent_dict[phase](features)
        loss = F.l1_loss(predicts, labels)
        self.optimizer_dict[phase].zero_grad()
        loss.backward()
        self.optimizer_dict[phase].step()
        print(f'{phase}-{loss.item()}')
        return loss.item()

    def generate_graph(self, phase):
        #incoming_lanes = ['E2TL_1', 'E2TL_2', 'N2TL_1', 'N2TL_2', 'S2TL_1', 'S2TL_2', 'W2TL_1', 'W2TL_2']
        #对于一个phase: gGGgrrgrrgrr
        #按照 N-E-S-W 的顺序
        # N2TL_(0,1,2)-E2TL_(0,1,2)-S2TL(0,1,2)-W2TL_(0,1,2) 
        #右车道是0，中间是1，左车道是2
        #state也是根据这个顺序
        u, v = torch.tensor([N_IN, N_IN, N_IN, E_IN, E_IN, E_IN, S_IN, S_IN, S_IN, W_IN, W_IN, W_IN]), torch.tensor([W_OUT, S_OUT, E_OUT, N_OUT, W_OUT, S_OUT, E_OUT, N_OUT, W_OUT, S_OUT, E_OUT, N_OUT])
        connected_edges_from = []
        connected_edges_to = []
        stuck_edges_from = []
        stuck_edges_to = []
        
        for i, s in enumerate(phase):
            #TODO: g和G的权重可能不同
            if s=='g' or s=='G':
                connected_edges_from.append(u[i])
                connected_edges_to.append(v[i])
            else:
                stuck_edges_from.append(u[i])
                stuck_edges_to.append(v[i])
        g = dgl.heterograph({
            ('road', 'connected', 'road'): (connected_edges_from, connected_edges_to),
            ('road', 'stuck', 'road'): (stuck_edges_from, stuck_edges_to)
        })
        # g = dgl.add_self_loop(g, etype=self.edge_type)
        return g