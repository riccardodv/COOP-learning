import numpy as np
import bandit
from collections import deque
import networkx as nx
import matplotlib.pyplot as pp

def run_coop_algo(ban, coop, T):
    r = np.array([])
    for t in range(T):
        coop.act()
        r = np.append(r, ban.total_regret())
    ########## SOME PRINT ######################################################
        # print("----> Round =", t+1)
        # print("eta = ", ban.eta)
        # print("total regret:", ban.total_regret())
        # print("EDGES:", ban.net_agents.edges)
        # print("optima arm:", arms_means[0])
        # for v in ban.net_agents.nodes:
        #     print("agent", v, "T:", ban.net_agents.nodes[v]['T'], "r:",np.dot(ban.subopt, ban.net_agents.nodes[v]['T']), "Tsub:", ban.t - ban.net_agents.nodes[v]['T'][0])
    ############################################################################
    return ban, coop, r/coop.Q


n = 2
f = 2
num_agents = 10
num_arms = 10
T = 1000
arms_means = 1/2 * np.ones(num_arms)
arms_means[0] = 0.3 #1/2 - np.sqrt(num_arms/T)
sample = 3
rr , rr_indepp = [], []
ban = bandit.Bandit(arms_means, num_agents, num_arms, n, f)
coop = bandit.COOP_algo(ban, T)
for s in range(sample):
    ban.restart_bandit()
    coop.restart_algo(ban, T)
    ban, coop, r = run_coop_algo(ban, coop, T)
    rr.append(r)
for s in range(sample):
    ban.restart_bandit()
    ban.net_agents.remove_edges_from(ban.net_agents.edges())
    coop.restart_algo(ban, T)
    ban, coop, r_indepp = run_coop_algo(ban, coop, T)
    rr_indepp.append(r_indepp)

rr = np.array(rr)
rr_indepp = np.array(rr_indepp)

means = np.mean(rr, axis=0)
errors = np.std(rr, axis=0)
means_indepp = np.mean(rr_indepp, axis=0)
errors_indepp = np.std(rr_indepp, axis=0)
x = [i for i in range(T)]


pp.plot(x, means, label = "COOPalgo", color='blue')
pp.fill_between(x, means - errors,means + errors,
                 color='blue', alpha=0.2)
pp.plot(x, means_indepp, label = "independent", color='red')
pp.fill_between(x, means_indepp - errors_indepp,means_indepp + errors_indepp,
                 color='red', alpha=0.2)
pp.title(f"n={n}, f={f}, n agents={num_agents}, n arms={num_arms}, subopt={arms_means[0]}")
pp.legend()
pp.show()
