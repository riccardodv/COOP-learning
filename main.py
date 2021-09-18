import numpy as np
import bandit
from collections import deque
# import importlib
# importlib.reload(bandit)

n = 1
f = 1
num_agents = 5
num_arms = 10
T = 10000
arms_means = 1/2*np.ones(num_arms)
arms_means
arms_means[0] = 0.1 #1/2 - np.sqrt(num_arms/T)
ban = bandit.Bandit(arms_means, num_agents, num_arms, n, f)
coop = bandit.COOP_algo(ban, T)
r = np.array([])
for t in range(T):
    coop.act()
    print("ban.total_regret()", ban.total_regret(),"|||",  ban.net_agents.nodes[0]['T'], np.dot(ban.subopt, ban.net_agents.nodes[0]['T']),
        "|||", ban.net_agents.nodes[1]['T'], np.dot(ban.subopt, ban.net_agents.nodes[1]['T']),
         "|||", ban.net_agents.nodes[2]['T'], np.dot(ban.subopt, ban.net_agents.nodes[2]['T']),
         "|||", ban.net_agents.nodes[3]['T'], np.dot(ban.subopt, ban.net_agents.nodes[3]['T']),
         "|||", ban.net_agents.nodes[4]['T'], np.dot(ban.subopt, ban.net_agents.nodes[4]['T'])

         )
    r = np.append(r, ban.total_regret())


r = r/coop.Q
import matplotlib.pyplot as pp
pp.plot(r)
pp.show()


# n = 1
# f = 1
# num_agents = 20
# num_arms = 2
# T = 50
# r = []
# arms_means = 1/2*np.ones(num_arms)
# ban = bandit.Bandit(arms_means, num_agents, num_arms, n, f)
# coop = bandit.COOP_algo(ban)
# for t in range(1,T+1):
#     coop.bandit.means[0] = 0.01 #1/2 - np.sqrt(1/5/t)
#     coop.T = t
#     f_var = coop.bandit.arms()
#     s_var = coop.bandit.net_agents.number_of_nodes()
#     coop.W = np.ones((f_var, s_var))
#     coop.P = np.ones((f_var, s_var))/f_var
#     coop.buffer = deque()
#     coop.bandit.subopt = coop.bandit.means - np.min(coop.bandit.means)
#     coop.bandit.t = 0
#     for v in coop.bandit.net_agents.nodes:
#         coop.bandit.net_agents.nodes[v]['new_losses'] = []
#         coop.bandit.net_agents.nodes[v]['message'] = np.zeros(coop.bandit.arms())
#         coop.bandit.net_agents.nodes[v]['w'] = np.zeros(coop.bandit.arms())
#         coop.bandit.net_agents.nodes[v]['T'] = np.zeros(coop.bandit.arms())
#         coop.bandit.net_agents.nodes[v]['S'] = np.zeros(coop.bandit.arms())
#     print("-----------------")
#     for s in range(t):
#         coop.act()
#         # if s % 5 == 0:
#         print(ban.total_regret(), ban.net_agents.nodes[2]['T'])
#     r.append(ban.total_regret())
# print(r )
# r = r/coop.Q
# import matplotlib.pyplot as pp
# pp.plot(r)
# pp.show()
