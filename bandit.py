import numpy as np
import random
import networkx as nx
from networkx.algorithms.approximation import independent_set
from collections import deque
import copy
import matplotlib.pyplot as pp

def ev_eta_fixed(K, T, af, aa, Q, f, n):
    return np.sqrt(np.log(K)/T/(af/(1-np.exp(-1))*(aa/Q+1)+f+n))

class Bandit:
  def __init__(self, means, A, K, n, f, p_ERa = 0.1, p_ERf = 0.1, q = 1, seed_f =41, seed_a=42):
    assert len(means) == K
    self.means = means
    self.subopt = self.means - np.min(self.means)
    self.net_feed = nx.erdos_renyi_graph(K, p_ERf, seed = seed_f)
    self.net_agents = nx.erdos_renyi_graph(A, p_ERa, seed = seed_a)
    self.t = 0
    for v in self.net_agents.nodes:
        self.net_agents.nodes[v]['new_ls'] = []
        self.net_agents.nodes[v]['m'] = np.zeros(self.arms())
        self.net_agents.nodes[v]['T'] = np.zeros(self.arms())
        self.net_agents.nodes[v]['S'] = np.zeros(self.arms())
        self.net_agents.nodes[v]['q'] = q
        self.activations = []
        self.f = f
        self.n = n
        self.A = A
        self.K = K
    

  def rounds(self):
    return self.t

  def arms(self):
    return len(self.means)

  def regret(self, v):
    return np.dot(self.subopt, self.net_agents.nodes[v]['T'])

  def total_regret(self):
    R = 0
    gen = (v for v in self.net_agents.nodes if self.activations[v]==1)
    for v in gen:
      R += self.regret(v)
    return R

  def activate_players(self):
    self.activations = []
    for v in self.net_agents.nodes:
      self.activations.append(np.random.binomial(1, self.net_agents.nodes[v]['q']))

  def play(self, arms):
    assert len(self.net_agents.nodes) == len(arms)
    # Compute rewards for this round
    rewards = np.random.binomial(1, self.means)
    self.t += 1
    # delete previous losses and prepare for new onens
    for v in self.net_agents.nodes:
      self.net_agents.nodes[v]['new_ls'] = []
    # Update when you are active and play:
    gen = (v for v in self.net_agents.nodes if self.activations[v]==1)
    for v in gen:
      message = self.net_agents.nodes[v]['m'] # messages ~ probabilities
      i = arms[v]
      reward = rewards[i]
      self.net_agents.nodes[v]['T'][i] += 1
      self.net_agents.nodes[v]['S'][i] += reward
      neigh_feed = [j for j in nx.single_source_shortest_path_length(self.net_feed, i, self.f).keys()]
      for j in neigh_feed:
        self.net_agents.nodes[v]['new_ls'].append((self.t, v, j, rewards[j], message[j]))
    # Update for all the messages coming from active agents:
    for v in self.net_agents.nodes:
      neigh_agents = [u for u in nx.single_source_shortest_path_length(self.net_agents, v, self.n).keys()]
      # print("neighbour of", v, "has elements", neigh_agents)
      for u in neigh_agents:
        if u != v:
          self.net_agents.nodes[v]['new_ls'] += self.net_agents.nodes[u]['new_ls']
    return 0

  def restart_bandit(self):
    self.t = 0
    for v in self.net_agents.nodes:
        self.net_agents.nodes[v]['new_ls'] = []
        self.net_agents.nodes[v]['m'] = np.zeros(self.arms())
        self.net_agents.nodes[v]['T'] = np.zeros(self.arms())
        self.net_agents.nodes[v]['S'] = np.zeros(self.arms())
        self.activations = []

class COOP_algo():
  def __init__(self, bandit, T=100):
    self.bandit = bandit
    f_var = self.bandit.arms()
    s_var = self.bandit.net_agents.number_of_nodes()
    self.W = np.ones((f_var, s_var))
    self.P = np.ones((f_var, s_var))/f_var
    self.T = T
    self.buffer = deque()
    self.alpha_feed = len(independent_set.maximum_independent_set(nx.power(self.bandit.net_feed, self.bandit.f)))
    self.alpha_agents = len(independent_set.maximum_independent_set(nx.power(self.bandit.net_agents, self.bandit.n)))
    self.Q = np.sum([self.bandit.net_agents.nodes[v]['q'] for v in self.bandit.net_agents.nodes])
    self.eta = ev_eta_fixed(self.bandit.K, self.T, self.alpha_feed, self.alpha_agents, self.Q, self.bandit.f, self.bandit.n)

  def update(self, eta):
    for v in range(len(self.bandit.net_agents.nodes)):
      for i in range(self.bandit.arms()):
        self.W[i,v] = self.W[i,v] * np.exp(-eta*self.ev_lhat(i,v))
    for v in range(len(self.bandit.net_agents.nodes)):
      self.P[:,v] = self.W[:,v]/np.linalg.norm(self.W[:,v], ord=1)
      # print("P = ", self.P[:,v])

  def update_buffer_and_predict(self):
    # Adaptive learning rate:
    # self.eta = 1/np.sqrt(self.bandit.t+1)
    # self.update(self.eta)
    # Fixed learning rate:
    self.update(self.eta)
    # print("eta = ", self.eta)
    self.buffer.append(copy.copy(self.P))
    # print("self.buffer: ", self.buffer)
    if self.bandit.t >= self.bandit.f + self.bandit.n:
      return self.buffer.popleft()
    else:
      return self.buffer[0]

  def ev_lhat(self, i, v, epsilon = 10**-20):
    new_losses = self.bandit.net_agents.nodes[v]['new_ls']
    if len(new_losses) != 0:
      loss_comps = {(j, reward) for s, u, j, reward, prob in new_losses} # maybe add prob????
      seen_comps = {j for s, u, j, reward, prob in new_losses}
      neigh_agents = [u for u in nx.single_source_shortest_path_length(self.bandit.net_agents, v, self.bandit.n).keys()]
      neigh_feed = [j for j in nx.single_source_shortest_path_length(self.bandit.net_feed, i, self.bandit.f).keys()]
      if i not in seen_comps:
        return 0.
      else:
        [l_i] = [l[1] for l in loss_comps if l[0]==i]
        prod = 1
        for u in neigh_agents:
          P = np.sum([self.P[j,u] for j in neigh_feed])
          I_P = 1 - self.bandit.net_agents.nodes[u]['q'] * P
          prod *= I_P
        b = 1 - prod + epsilon
        # print("b=",b, "|| l_i / b = ",l_i / b, "|| eta * l_i / b = ",l_i / b*self.eta)
        return l_i / b
    else:
      return 0.

  def sample(self):
    probs = self.update_buffer_and_predict()
    arms = [a for a in range(len(self.bandit.net_agents.nodes))]
    for v, act in enumerate(self.bandit.activations):
      prob = probs[:,v]
      # print("NEW LOSSES AVAILABLE: ", self.bandit.net_agents.nodes[v]['new_ls'])
      # print("PREDICTION PROB: ", prob)
      self.bandit.net_agents.nodes[v]['m'] = prob

      if act == 1:
        [arms[v]] = np.random.choice(list(self.bandit.net_feed.nodes), 1, p = prob)
      else:
        arms[v] = None
    # print("arms played:", arms)
    return arms

  def act(self):
    self.bandit.activate_players()
    self.bandit.play(self.sample())

  def restart_algo(self, bandit, T):
    self.bandit = bandit
    f_var = self.bandit.arms()
    s_var = self.bandit.net_agents.number_of_nodes()
    self.W = np.ones((f_var, s_var))
    self.P = np.ones((f_var, s_var))/f_var
    self.T = T
    self.buffer = deque()
    self.alpha_feed = len(independent_set.maximum_independent_set(nx.power(self.bandit.net_feed, self.bandit.f)))
    self.alpha_agents = len(independent_set.maximum_independent_set(nx.power(self.bandit.net_agents, self.bandit.n)))
    self.Q = np.sum([self.bandit.net_agents.nodes[v]['q'] for v in self.bandit.net_agents.nodes])
    self.eta = ev_eta_fixed(self.bandit.K, self.T, self.alpha_feed, self.alpha_agents, self.Q, self.bandit.f, self.bandit.n)
