import numpy as np
import scipy.linalg as la
import scipy.stats as sps
from collections import Counter


# Here we define the a Jackson Network Node.
def make_callable(x):
    return x if callable(x) else lambda _: x

class Node:
    '''A Node object in a Jackson network simulates a queue with an exponential
    service time.'''
    def __init__(self, mu, n, r, dt=1):
        '''
        mu     : Service rate (average vehicles per time), constant or a function
                 of number of vehicles in node
        n      : Initial number of vehicles in the node
        r      : Routing probability from node id to probability
        dt     : Timestep
        '''
        self.r = r
        self.check_r()
        self.mu = make_callable(mu)
        self.n = n  # Number in queue

    def check_r(self):
        assert abs(sum(self.r.values()) - 1.0) < 1e-15, 'Routing probabilities do not sum to 1!'

    def add(self, n):
        self.n += n

    def route_to(self):
        '''Samples a destination from the routing probability distribution'''
        p, s = np.random.sample(), 0
        for id, prob in self.r.items():
            s += prob
            if p < s: return id
        print p, s, id, self.r

class Network:
    def __init__(self, graph):
        self.graph = graph
        self.t = 0

    def to_matrix(self):
        n = len(self.graph)
        res = []
        for i in range(n):
            for j in range(n):
                res.append(self.graph[i].r.get(j) or 0)
        return np.reshape(res, (n, n)).T

    def tick(self):
        '''Simulates one tick of a network'''
        # Get total rates
        rates = [node.mu(node.n) for node in self.graph.values()]
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
        self.graph[update].n -= 1

        # Pick destination
        dest_dist = sps.rv_discrete(values=zip(*self.graph[update].r.items()))
        self.graph[dest_dist.rvs()].n += 1


    def get_counts(self):
        return zip(*[(i, node.n) for i, node in self.graph.items()])



def full_network(k, lam, n):
    '''Creates a network of k nodes (fully connected graph) with equal routing
    probabilities and service times, ignoring travel times. Each node starts
    with n vehicles.'''
    prob = 1.0 / (k-1) # Equal probability to go to each node
    graph = {}
    for i in range(k):
        graph[i] = Node(lam, n, {j: prob for j in range(k) if j != i})
    return Network(graph)

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
