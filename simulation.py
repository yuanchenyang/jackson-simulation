import numpy as np
import scipy.linalg as la
import scipy.stats as sps
import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter


# Here we define the a Jackson Network Node.
def make_callable(x):
    return x if callable(x) else lambda _: x

# Checks if r is a probability distribution
def check_dist(r):
    for x, px in r.items():
        assert px >= 0, 'Not a distribution!'
    assert abs(sum(r.values()) - 1.0) < 1e-15, 'Not a distribution!'

class Node:
    '''A Node object in a Jackson network simulates a queue with an exponential
    service time.'''
    def __init__(self, id, n, mu):
        '''
        id     : Hashable id given to the node
        mu     : Service rate (average vehicles per time), a function of the
                 number of vehicles in node
        n      : Initial number of vehicles in the node
        '''
        self.id = id
        self.n = n
        self.mu = mu

    def add(self, n):
        self.n += n

    def service_rate(self):
        '''Returns the current service rate of this node'''
        raise NotImplementedError

    def route_to(self):
        '''Randomly samples a destination (node id) from the routing probability
        distribution'''
        raise NotImplementedError

class RoadNode(Node):
    def __init__(self, id, n, mu, rid):
        '''
        rid    : Single ID to route to
        mu     : rate of service for each car
        '''
        Node.__init__(self, id, n, mu)
        self.rid = rid

    def service_rate(self):
        return self.mu * self.n

    def route_to(self):
        return rid

class StationNode(Node):
    def __init__(self, id, n, mu, r):
        '''
        r     : Routing probability, dict from node id to probability
        '''
        Node.__init__(self, id, n, mu)
        check_dist(r)
        self.r = r
        self.rdist = sps.rv_discrete(values=zip(*r.items()))

    def service_rate(self):
        return self.mu

    def route_to(self):
        return self.rdist.rvs(1)

class Network:
    def __init__(self, n, lam, T, p, k):
        '''Creates a network of n nodes with the following parameters

        n   : Number of nodes
        lam : lam[i] is the arrival rate at station node i
        T   : T[i][j] is the average travel time from node i to node j
        p   : p[i][j] is the routing probability from node i to node j
        k   : k[i] is the number of vehicles station i starts with
        '''
        assert len(k) == len(p) == len(T) == len(lam) == n, 'Seq lengths incorrect!'
        for a, b in zip(T, p):
            assert len(a) == len(b) == n, 'Sub-seq lengths incorrect!'

        self.graph = {}
        self.t = 0
        for i in range(n):
            r = {}
            for j in range(n):
                if i != j:
                    rn_name = '{}->{}'.format(i, j)
                    self.add_node(RoadNode(rn_name, 0, T[i][j], j))
                    r[rn_name] = p[i][j]
            self.add_node(StationNode(str(i), k[i], lam[i], r))

    def add_node(self, node):
        self.graph[node.id] = node

    def add_attack(self, i, psi, alpha):
        ''' Adds attack to node i with arrival rate psi and routing probability
        alpha

        psi  : A number representing the arrival rate of attackers
        alpha: A dictionary where alpha[j] is the attackers' routing probability
               rate from node i to node i
        '''
        node = self.graph[i]
        assert isinstance(node, StationNode), 'Can only attack stations'
        newlam = node.mu + psi
        for j in node.r:
            node.r[j] = (alpha[j] * psi +  node.mu * node.r[j]) / newlam
        node.mu = newlam

    def to_matrix(self):
        n = len(self.graph)
        res = []
        for i in range(n):
            for j in range(n):
                res.append(self.graph[i].r.get(j) or 0)
        return np.reshape(res, (n, n)).T

    def as_networkx(self):
        G = nx.DiGraph()
        for nodeid, node in self.graph.items():
            G.add_node(nodeid, {'label': 'n={}, mu={}'\
                                .format(node.n, round(node.service_rate(), 3))})

        for nodeid, node in self.graph.items():
            if isinstance(node, RoadNode):
                G.add_edge(nodeid, node.rid, {'label': 'p=1'})
            else:
                for nodeid_to, prob in node.r.items():
                    G.add_edge(nodeid, nodeid_to,
                               {'label': 'p={}'.format(round(prob, 4))})
        return G

    def write_graphviz(self, filename):
        nx.write_dot(self.as_networkx(), filename)

    def tick(self):
        '''Simulates one tick of a network'''
        # Get total rates
        rates = [node.service_rate() for node in self.graph.values()]
        total_rates = float(sum(rates))
        # Update time
        dt = np.random.exponential(1/total_rates)
        self.t += dt

        # Get node to update
        normalized_rates = [r/total_rates for r in rates]
        dist = sps.rv_discrete(values=(range(len(rates)), normalized_rates))
        update = dist.rvs()

        # Update node if it isn't empty
        if self.graph[update].n == 0:
            return
        self.graph[update].add(-1)

        # Pick destination
        dest = self.graph[update].route_to()
        self.graph[dest].add(1)


    def get_counts(self):
        return zip(*[(i, node.n) for i, node in self.graph.items()])

def full_network(n, lam, T, k):
    ''' Same routing probabilities, constant lam and t. '''
    T = [[T if i != j else 0 for i in range(n)] for j in range(n)]
    p = [[1/float(n-1) if i != j else 0 for i in range(n)] for j in range(n)]
    return Network(n, [lam] * n, T, p, [k] * n)

def linear_network(k, psi, lam, n):
    '''A network of nodes, with a linear virtual passenger chain from node i
    to node i+1, with service rate psi.'''
    lam, psi = make_callable(lam), make_callable(psi)
    nw = full_network(k, lam, n=n)
    for i, node in nw.graph.items():
        if i == k-1: # Don't alter last node in chain
            continue
        node.mu = lambda x: psi(i) + lam(i)
        for j in node.r:
            newr = float(lam(i) * node.r[j]) / (lam(i) + psi(i))
            if j == i+1:
                newr += float(psi(i)) / (lam(i) + psi(i))
            node.r[j] = newr
        node.check_r()
    return nw

def grid_network_3x3():
    pass


if __name__ == '__main__':
    N = linear_network(5, 0.1, 1, 15)
    for _ in range(1000):
        N.tick()
    N.write_graphviz('out.dot')
