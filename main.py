import numpy as np
import bandit
from collections import deque
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pp
import multiprocessing as mp

def run_algo(ban, coop, T):
    r = np.array([])
    for t in range(T):
        coop.act()
        r = np.append(r, ban.total_regret())
    ########## SOME PRINT ######################################################
        # print("----> Round =", t+1)
        print("eta = ", coop.eta)
        # print("total regret:", ban.total_regret())
        # print("EDGES:", ban.net_agents.edges)
        # print("optima arm:", arms_means[0])
        # for v in ban.net_agents.nodes:
        #     print("agent", v, "has neighbours:",[u for u in ban.net_agents.neighbors(v)],
        #     "T:", ban.net_agents.nodes[v]['T'], "r:",np.dot(ban.subopt, ban.net_agents.nodes[v]['T']), "Tsub:", ban.t - ban.net_agents.nodes[v]['T'][0])
    ############################################################################
    return ban, coop, r/coop.Q

def UB(x_list, d, K, alpha_feed, alpha_agents):
    f = lambda x: d+2*np.sqrt(np.log(K)*x*(alpha_feed/(1-np.exp(-1))*(alpha_agents/coop.Q+1)+d))
    l = [x for x in map(f, x_list)]
    return l

# n = 2
# f = 2
# num_agents = 10
# num_arms = 10
# T = 500
# prob_ER_a = 0.3; prob_ER_f = 0.2
# arms_means = 1/2 * np.ones(num_arms)
# arms_means[0] = 1/2 - np.sqrt(num_arms/T)
# sample = 1
#
# rr , rr_indepp = [], []
# ban = bandit.Bandit(arms_means, num_agents, num_arms, n, f, prob_ER_a, prob_ER_f)
# coop = bandit.COOP_algo(ban, T)
# for s in range(sample):
#     ban.restart_bandit()
#     coop.restart_algo(ban, T)
#     ban, coop, r = run_algo(ban, coop, T)
#     rr.append(r)
# for s in range(sample):
#     ban.restart_bandit()
#     ban.net_agents.remove_edges_from(ban.net_agents.edges())
#     coop.restart_algo(ban, T)
#     ban, coop, r_indepp = run_algo(ban, coop, T)
#     rr_indepp.append(r_indepp)
#
# rr = np.array(rr)
# rr_indepp = np.array(rr_indepp)
#
# means = np.mean(rr, axis=0)
# errors = np.std(rr, axis=0)
# means_indepp = np.mean(rr_indepp, axis=0)
# errors_indepp = np.std(rr_indepp, axis=0)
# x = [i for i in range(T)]
# bound = UB(x, ban.f+ban.n, ban.K, coop.alpha_feed, coop.alpha_agents)

n = 2
f = 2
q = 1
num_agents = 10
num_arms = 10
T = 1000
prob_ER_a = 0.3; prob_ER_f = 0.2
arms_means = 1/2 * np.ones(num_arms)
arms_means[0] = 0.2 #1/2 - np.sqrt(num_arms/T)
sample = 10

def run_indep_coop(arms_means, num_agents, num_arms, n, f, prob_ER_a, prob_ER_f, T):
# rr , rr_indepp = [], []
    ban = bandit.Bandit(arms_means, num_agents, num_arms, n, f, prob_ER_a, prob_ER_f, q)
    coop = bandit.COOP_algo(ban, T)
    ban, coop, r = run_algo(ban, coop, T)
    ban.restart_bandit()
    ban.net_agents.remove_edges_from(ban.net_agents.edges())
    coop.restart_algo(ban, T)
    ban, coop, r_indepp = run_algo(ban, coop, T)
    return r, r_indepp

if __name__ == "__main__":
    print("cpus number =", mp.cpu_count())
    pool = mp.Pool(4) #(mp.cpu_count())
    it = [(arms_means, num_agents, num_arms, n, f, prob_ER_a, prob_ER_f, T) for s in range(sample)]
    results = pool.starmap(run_indep_coop, it)
    pool.close()
    pool.join()
    results = np.array(results)
    r = results[:,0]
    r_indepp = results[:,1]

    means = np.mean(r, axis=0)
    errors = np.std(r, axis=0)
    means_indepp = np.mean(r_indepp, axis=0)
    errors_indepp = np.std(r_indepp, axis=0)
    x = [i for i in range(T)]
    # bound = UB(x, n+f, num_arms, coop.alpha_feed, coop.alpha_agents)
    pp.figure(figsize=(10, 6))
    pp.plot(x, means, label = "COOPalgo", color='blue')
    pp.fill_between(x, means - errors,means + errors, color='blue', alpha=0.2)
    pp.plot(x, means_indepp, label = "independent", color='red')
    pp.fill_between(x, means_indepp - errors_indepp,means_indepp + errors_indepp, color='red', alpha=0.2)
    # pp.plot(x, bound, label = "theory", color='black')
    tit = f"n={n}, f={f}, n agents={num_agents}, n arms={num_arms}, bias0={arms_means[0]:.3}"
    tit += f", q={q}, samples={sample}, ER_a={prob_ER_a}, ER_f={prob_ER_f}"
    pp.title(tit)
    pp.legend()
    pp.savefig('fig.pdf', dpi=600)
