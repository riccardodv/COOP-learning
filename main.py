import numpy as np
import bandit
from collections import deque
import networkx as nx

n = 2
f = 2
num_agents = 10
num_arms = 5
T = 10000
arms_means = 1/2 * np.ones(num_arms)
arms_means[0] = 1/2 - np.sqrt(num_arms/T)


# Run cooperative algorithm:
ban = bandit.Bandit(arms_means, num_agents, num_arms, n, f)
coop = bandit.COOP_algo(ban, T)
r = np.array([])
for t in range(T):
    print("EDGES: ", ban.net_agents.edges)
    print("----> Round =", t+1)
    print("1/2 - np.sqrt(num_arms/T) = ", 1/2 - np.sqrt(num_arms/T))
    coop.act()
    print("ban.total_regret()", ban.total_regret())
    for v in ban.net_agents.nodes:
        print("agent", v, "T:", ban.net_agents.nodes[v]['T'], "r:",np.dot(ban.subopt, ban.net_agents.nodes[v]['T']), "Tsub:", ban.t - ban.net_agents.nodes[v]['T'][0])
    r = np.append(r, ban.total_regret())

print("arms_means[0] = ",arms_means[0])
print("coop.Q",coop.Q, "|| coop.T",coop.T, "|| alpha_feed", coop.alpha_feed, "|| alpha_agents", coop.alpha_agents)


# Run algo with agents playing independently:
ban.restart_bandit()
ban.net_agents.remove_edges_from(ban.net_agents.edges())
coop.restart_algo(ban, T)
r_indepp = np.array([])
for t in range(T):
    print("EDGES: ", ban.net_agents.edges)
    print("----> Round =", t)
    print("1/2 - np.sqrt(num_arms/T) = ", 1/2 - np.sqrt(num_arms/T))
    coop.act()
    print("ban.total_regret()", ban.total_regret())
    for v in ban.net_agents.nodes:
        print("agent", v, "T:", ban.net_agents.nodes[v]['T'], "r:",np.dot(ban.subopt, ban.net_agents.nodes[v]['T']), "Tsub:", ban.t - ban.net_agents.nodes[v]['T'][0])
    r_indepp = np.append(r_indepp, ban.total_regret())


# print stuff and plot
print("arms_means[0] = ",arms_means[0])
print("coop.Q",coop.Q, "|| coop.T",coop.T, "|| alpha_feed", coop.alpha_feed, "|| alpha_agents", coop.alpha_agents)

r = r/coop.Q
r_indepp = r_indepp/coop.Q
import matplotlib.pyplot as pp
pp.plot(r,'b--', label = "COOPalgo")
pp.plot(r_indepp, 'r--', label = "non-cooperative")
pp.legend()
pp.show()
