import sys, subprocess, os
import inspect

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'],'tools')
    sys.path.append(tools)
    import sumolib
else:
    sys.exit("please declare environment viriable 'SUMO_HOME'")

import numpy as np

class RoadnetReader:
    def __init__(self,roadnet):

        self.net = sumolib.net.readNet(os.path.join('roadnet', roadnet+'.net.xml'))
        self.edge_data = self._get_edge_data(self.net)
        self.lane_data = self._get_lane_data(self.net)
        self.node_data, self.intersection_data = self._get_node_data(self.net)
        self.origins = self._get_origin_edges()
        self.destinaitons = self._get_destination_edges()

    def get_net_data(self):
        return {'lane':self.lane_data, 'edge':self.edge_data, 'origin':self.origins, 'destination':self.destinaitons, 'node':self.node_data, 'inter':self.intersection_data}

    def _get_edge_data(self, net):
        edges = net.getEdges()
        edge_data = {str(edge.getID()):{} for edge in edges}

        for edge in edges:
            edge_ID = str(edge.getID())
            edge_data[edge_ID]['lanes'] = [str(lane.getID()) for lane in edge.getLanes()]
            edge_data[edge_ID]['length'] = float(edge.getLength())
            edge_data[edge_ID]['outgoing'] = [str(out.getID()) for out in edge.getOutgoing()]
            edge_data[edge_ID]['noutgoing'] = len(edge_data[edge_ID]['outgoing'])
            edge_data[edge_ID]['nlanes'] = len(edge_data[edge_ID]['lanes'])
            edge_data[edge_ID]['incoming'] = [str(inc.getID()) for inc in edge.getIncoming()]
            edge_data[edge_ID]['outnode'] = str(edge.getFromNode().getID())
            edge_data[edge_ID]['incnode'] = str(edge.getToNode().getID())
            edge_data[edge_ID]['speed'] = float(edge.getSpeed())
        
        return edge_data

    def _get_destination_edges(self):
        next_edges = { e:0 for e in self.edge_data }
        for e in self.edge_data:
            for next_e in self.edge_data[e]['incoming']:
                next_edges[next_e] += 1
                                                                 
        destinations = [ e for e in next_edges if next_edges[e] == 0]
        return destinations

    def _get_origin_edges(self):

        next_edges = {e:0 for e in self.edge_data}
        for e in self.edge_data:
            for next_e in self.edge_data[e]['outgoing']:
                next_edges[next_e] += 1

        origins = [e for e in next_edges if next_edges[e]==0]
        return origins

    def _get_lane_data(self, net):
        lane_ids = []
        for edge in self.edge_data:
            lane_ids.extend(self.edge_data[edge]['lanes'])

        lanes = [net.getLane(lane) for lane in lane_ids]
        lane_data = {id:{} for id in lane_ids}
        for lane in lanes:
            lane_id = lane.getID()
            lane_data[lane_id]['length'] = lane.getLength()
            lane_data[lane_id]['speed'] = lane.getSpeed()
            lane_data[lane_id]['edge'] = str(lane.getEdge().getID())
            lane_data[lane_id]['outgoing'] = {}

            moveid = []
            #.getOutgoing return a Connection type
            for conn in lane.getOutgoing():
                out_id = str(conn.getToLane().getID())
                lane_data[lane_id]['outgoing'][out_id] = {'dir':str(conn.getDirection()),'index':conn.getTLLinkIndex()}
                moveid.append(str(conn.getDirection()))
            lane_data[lane_id]['movement'] = ''.join(sorted(moveid))
            #create empty list for incoming lanes

            lane_data[lane_id]['incoming']=[]
        for lane in lane_data:
            for inc in lane_data:
                if lane == inc:
                    continue
                else:
                    if inc in lane_data[lane]['outgoing']:
                        lane_data[inc]['incoming'].append(lane)

        return lane_data

    def _get_node_data(self, net):
        nodes = net.getNodes()
        node_data = {str(node.getID()):{} for node in nodes}

        for node in nodes:
            node_ID = node.getID()
            node_data[node_ID]['incoming'] = set(str(edge.getID()) for edge in node.getIncoming())
            node_data[node_ID]['outgoing'] = set(str(edge.getID()) for edge in node.getOutgoing())
            node_data[node_ID]['tlsindex'] = { conn.getTLLinkIndex():str(conn.getFromLane().getID()) for conn in node.getConnections()}
            node_data[node_ID]['tlsindexdir'] = { conn.getTLLinkIndex():str(conn.getDirection()) for conn in node.getConnections()}

            if node_ID == '-13968':
                missing = []
                negative = []
                for i in range(len(node_data[node_ID]['tlsindex'])):
                    if i not in node_data[node_ID]['tlsindex']:
                        missing.append(i)

                for k in node_data[node_ID]['tlsindex']:
                    if k < 0  :
                        negative.append(k)
              
                for m,n in zip(missing, negative):
                    node_data[node_ID]['tlsindex'][m] = node_data[node_ID]['tlsindex'][n]
                    del node_data[node_ID]['tlsindex'][n]
                    #for index dir
                    node_data[node_ID]['tlsindexdir'][m] = node_data[node_ID]['tlsindexdir'][n]
                    del node_data[node_ID]['tlsindexdir'][n]

            #get XY coords
            pos = node.getCoord()
            node_data[node_ID]['x'] = pos[0]
            node_data[node_ID]['y'] = pos[1]

        
        intersection_data = {str(node):node_data[node] for node in node_data if "traffic_light" in net.getNode(node).getType()}
        for inter in intersection_data:
            sorted_tlsindex = {}
            for key in sorted(intersection_data[inter]['tlsindex'].keys()):
                sorted_tlsindex[key] = intersection_data[inter]['tlsindex'][key]
            intersection_data[inter]['tlsindex'] = sorted_tlsindex
        return node_data, intersection_data