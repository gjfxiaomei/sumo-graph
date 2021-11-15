import torch
import torch.nn as nn
import dgl.nn as dglnn
from dgl.utils import expand_as_pair

class GraphNodePrediction(nn.module):
    def __init__(self, 
                 in_featrues,
                 out_features,
                 activation=None) -> None:
        super(GraphNodePrediction).__init__()
        self.layers = nn.ModuleDict()

        self.layers['message_module'] = Message_Module()
        self.layers['aggregation_module'] = Aggregation_Module()
    
    def forward(self, graph, device, num_propagations):
        #预测当前graph在phase结束时的节点特征
        for _ in range(num_propagations):
            self.layers['message_module'].propagate(graph, device)
            self.layers['aggregation_module'].aggregate(graph, device)

class Message_Module(nn.Module):
    def __init__(self) -> None:
        super(Message_Module, self).__init__()
        pass

    def propagate(self, graph, device):
        def message_func(edges):
            return {'msg': edges.src['queue'] * edges.data['weight']}

        def reduce_func(nodes):
            agg_msg = torch.sum(nodes.mailbox['msg'], dim=1)
            return {'agg_msg': agg_msg}

        def apply_func(nodes):
            pass

        graph.update_all(message_func=message_func, reduce_func=reduce_func, apply_func=apply_func)

class Aggregation_Module(nn.Module):
    def __init__(self) -> None:
        super(Aggregation_Module, self).__init__()
        pass
    
    def aggregate(self, graph, device):

        def message_func(edges):
            pass

        def reduce_func(nodes):
            pass

        def apply_func(nodes):
            agg_msg = nodes.data['agg_msg']
            return {'hid' : agg_msg.to(device)}
        
        graph.update_all(message_func=message_func, reduce_func=reduce_func, apply_func=apply_func)