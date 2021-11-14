import numpy as np
import random
import networkx as nx
from networkx.algorithms.approximation import independent_set
from collections import deque
import copy


class Bandit:
  def __init__(self, means, A, K, n, f, p_ERa, p_ERf , q, seed_a, seed_f):
    assert len(means) == K
    self.means = means
    self.subopt = self.means - np.min(self.means)
    self.net_feed = nx.erdos_renyi_graph(K, p_ERf, seed = seed_f)
    self.net_agents = nx.erdos_renyi_graph(A, p_ERa, seed = seed_a)
    self.t = 0
    self.activations = []
    self.f = f
    self.n = n
    self.A = A
    self.K = K
    self.neigh_agents = [0]*self.A
    self.neigh_feed = [0]*self.K

    for v in self.net_agents.nodes:
        self.net_agents.nodes[v]['new_ls'] = []
        self.net_agents.nodes[v]['my_ls'] = []
        self.net_agents.nodes[v]['m'] = np.zeros(self.arms())
        self.net_agents.nodes[v]['T'] = np.zeros(self.arms())
        self.net_agents.nodes[v]['S'] = np.zeros(self.arms())
        self.net_agents.nodes[v]['q'] = q
        self.neigh_agents[v] = [u for u in nx.single_source_shortest_path_length(self.net_agents, v, self.n).keys()]
    for i in self.net_feed.nodes:
        self.neigh_feed[i] = [j for j in nx.single_source_shortest_path_length(self.net_feed, i, self.f).keys()]

  def rounds(self):
    return self.t

  def arms(self):
    return len(self.means)

  def regret(self, v):
    return np.dot(self.subopt, self.net_agents.nodes[v]['T'])

  def total_regret(self):
    R = 0; Q = 0
    for v in self.net_agents.nodes:
      Q += self.net_agents.nodes[v]['q']
      R += self.regret(v)
    return R/Q

  def activate_players(self):
    self.activations = []
    for v in self.net_agents.nodes:
      rng = np.random.RandomState(self.t*self.A+v)
      self.activations.append(rng.binomial(1, self.net_agents.nodes[v]['q']))

  def play(self, arms):
    # Compute losses for this round
    assert len(self.net_agents.nodes) == len(arms)
    rng = np.random.RandomState(self.t)
    losses = rng.binomial(1, self.means)
    self.t += 1
    # delete previous losses and prepare for new onens
    for v in self.net_agents.nodes:
      self.net_agents.nodes[v]['new_ls'] = []
      self.net_agents.nodes[v]['my_ls'] = []
    # Update when you are active and play:
    gen = (v for v in self.net_agents.nodes if self.activations[v]==1)
    for v in gen:
      message = self.net_agents.nodes[v]['m'] # messages ~ probabilities
      i = arms[v]
      loss = losses[i]
      self.net_agents.nodes[v]['T'][i] += 1
      self.net_agents.nodes[v]['S'][i] += loss
      for j in self.neigh_feed[i]:
        self.net_agents.nodes[v]['my_ls'].append((self.t, v, j, losses[j], message[j]))
    # Update for all the messages coming from active agents:
    for v in self.net_agents.nodes:
      for u in self.neigh_agents[v]:
        self.net_agents.nodes[v]['new_ls'] += self.net_agents.nodes[u]['my_ls']
    return 0


class COOP_algo():
  def __init__(self, bandit, T=100):
    self.bandit = bandit
    self.W = np.ones((self.bandit.K, self.bandit.A))
    self.P = np.ones((self.bandit.K, self.bandit.A))/self.bandit.K
    self.T = T
    self.buffer = deque()
    self.alpha_feed = self.bandit.K if self.bandit.f == 0 else len(independent_set.maximum_independent_set(nx.power(self.bandit.net_feed, self.bandit.f)))
    self.alpha_agents = self.bandit.A if self.bandit.n == 0 else len(independent_set.maximum_independent_set(nx.power(self.bandit.net_agents, self.bandit.n)))
    self.Q = np.sum([self.bandit.net_agents.nodes[v]['q'] for v in self.bandit.net_agents.nodes])
    self.eta = np.ones(self.bandit.A)
    self.doubT = {'r' : np.ones(self.bandit.A)*(np.floor(np.log2(np.log(self.bandit.K)))+1), 'X' : np.zeros(self.bandit.A), 'c' : np.zeros(self.bandit.A)}

  def ev_eta_fixed(self, K, T, af, aa, Q, f, n):
      return np.sqrt(np.log(K)/T/(af/(1-np.exp(-1))*(aa/Q+1)+f+n))

  def ev_X(self, v, epsilon = 10**-20):
    d = self.bandit.n+self.bandit.f
    if self.doubT['c'][v] > d:
      bs = np.array([self.ev_b(i,v) for i in range(self.bandit.K)])
      x = d+np.sum(self.P[:,v]/bs)
      return x
    else:
      return 0.

  def ev_b(self, i, v, epsilon = 10**-20):
    new_losses = self.bandit.net_agents.nodes[v]['new_ls']
    prod = 1
    for u in self.bandit.neigh_agents[v]:
      P = np.sum([self.P[j,u] for j in self.bandit.neigh_feed[i]])
      I_P = 1 - self.bandit.net_agents.nodes[u]['q'] * P
      prod *= I_P
    b = 1 - prod + epsilon
    return b

  def update(self, eta):
    eta_m = np.array([eta[v] for v in range(self.bandit.A)]).reshape((1,self.bandit.A))
    lhat_m = np.array([[self.ev_lhat(i,v) for v in range(self.bandit.A)] for i in range(self.bandit.K)])
    self.W *= np.exp(-eta_m*lhat_m)
    for v in range(len(self.bandit.net_agents.nodes)):
      self.P[:,v] = self.W[:,v]/np.linalg.norm(self.W[:,v], ord=1)

  def update_buffer_and_predict(self, lr):
    if lr == 'adaptive':
      self.eta.fill(1/np.sqrt(self.bandit.t+1))
    elif lr == 'fixed':
      self.eta.fill(self.ev_eta_fixed(self.bandit.K, self.T, self.alpha_feed, self.alpha_agents, self.Q, self.bandit.f, self.bandit.n))
    elif lr == 'dt':
      if self.bandit.t==0:
        self.eta.fill(np.sqrt(np.log(self.bandit.K)/(2**self.doubT['r'][0])))
      for v in range(self.bandit.A):
        if self.doubT['X'][v] <= 2**self.doubT['r'][v]:
          self.doubT['c'][v] += 1
          self.doubT['X'][v] += self.ev_X(v)
        else:
          self.doubT['r'][v] += 1
          self.eta[v] = np.sqrt(np.log(self.bandit.K)/(2**self.doubT['r'][v]))
          self.doubT['c'][v] = 0
          self.doubT['X'][v] = 0
    else:
      assert False
    self.update(self.eta)
    self.buffer.append(copy.copy(self.P))
    if self.bandit.t >= self.bandit.f + self.bandit.n:
      return self.buffer.popleft()
    else:
      return self.buffer[0]

  def ev_lhat(self, i, v, epsilon = 10**-20):
    new_losses = self.bandit.net_agents.nodes[v]['new_ls']
    if len(new_losses) != 0:
      loss_comps = {(j, loss) for s, u, j, loss, prob in new_losses}
      seen_comps = {j for s, u, j, loss, prob in new_losses}
      if i not in seen_comps:
        return 0.
      else:
        [l_i] = [l[1] for l in loss_comps if l[0]==i]
        prod = 1
        for u in self.bandit.neigh_agents[v]:
          P = np.sum([self.P[j,u] for j in self.bandit.neigh_feed[i]])
          I_P = 1 - self.bandit.net_agents.nodes[u]['q'] * P
          prod *= I_P
        b = 1 - prod + epsilon
        return l_i / b
    else:
      return 0.

  def sample(self, lr):
    probs = self.update_buffer_and_predict(lr)
    arms = [a for a in range(len(self.bandit.net_agents.nodes))]
    for v, act in enumerate(self.bandit.activations):
      prob = probs[:,v]
      self.bandit.net_agents.nodes[v]['m'] = prob
      if act == 1:
        [arms[v]] = np.random.choice(list(self.bandit.net_feed.nodes), 1, p = prob)
      else:
        arms[v] = None
    return arms

  def act(self, lr):
    self.bandit.activate_players()
    self.bandit.play(self.sample(lr))
