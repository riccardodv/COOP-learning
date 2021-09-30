import numpy as np
import bandit
from collections import deque
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pp
import multiprocessing as mp

def UB(x_list, d, K, alpha_feed, alpha_agents):
    f = lambda x: d+2*np.sqrt(np.log(K)*x*(alpha_feed/(1-np.exp(-1))*(alpha_agents/coop.Q+1)+d))
    l = [x for x in map(f, x_list)]
    return l

def run_algo(ban, coop, T):
    r = np.array([])
    for t in range(T):
        coop.act()
        r = np.append(r, ban.total_regret())
        ###### SOME PRINT ######################################################
        if (t+1) % 1 == 0:
            print("-----> Round:", t+1)
            print("eta =", coop.eta, "|| T =",coop.T, "|| alpha_f =", coop.alpha_feed,
                    "|| alpha_a =", coop.alpha_agents,"|| n =",coop.bandit.n, "|| f =",coop.bandit.f)
            # print("total regret:", ban.total_regret())
            # print("edges feedback network:", ban.net_feed.edges)
            # print("edges agent network:", ban.net_agents.edges)
            # print("optima arm:", arms_mean[0])
            # for v in ban.net_agents.nodes:
            #     print("agent", v, "has neighbours:",[u for u in ban.net_agents.neighbors(v)],
            #     "T:", ban.net_agents.nodes[v]['T'], "r:",np.dot(ban.subopt, ban.net_agents.nodes[v]['T']), "Tsub:", ban.t - ban.net_agents.nodes[v]['T'][0])
        ########################################################################
    return ban, coop, r/coop.Q

def run_algo_COOPAndIndep(arms_mean, A, K, n, f, p_ERa, p_ERf, T):
    ban = bandit.Bandit(arms_mean, A, K, n, f, p_ERa, p_ERf, q)
    coop = bandit.COOP_algo(ban, T)
    ban, coop, r = run_algo(ban, coop, T)
    ban.restart_bandit()
    ban.net_agents.remove_edges_from(ban.net_agents.edges())
    coop.restart_algo(ban, T)
    ban, coop, r_indepp = run_algo(ban, coop, T)
    return r, r_indepp

n = 2; f = 2
A = 20; K = 20
q = 1/A
T = 1000
p_ERa = 0.1; p_ERf = 0.1
seed_f =41; seed_a=43
arms_mean = 1/2 * np.ones(K)
arms_mean[0] = 0. #1/2 - np.sqrt(K/T)
sample = 10

# # Draw networks
# w = 6; h = 6; d = 300;
# pp.figure(figsize=(w, h), dpi=d)
# net_feed = nx.erdos_renyi_graph(K, p_ERf, seed = seed_f)
# nx.draw_networkx(net_feed, pos=nx.spring_layout(net_feed), node_color ="red", alpha=0.9)
# pp.axis ("off")
# # net_agents = nx.erdos_renyi_graph(A, p_ERa, seed = seed_a)
# # nx.draw_networkx(net_agents,  pos=nx.spring_layout(net_agents), node_color ="blue", alpha=0.9)
# # pp.axis ("off")
# # pp.show()
# pp.savefig("out.pdf")


if __name__ == "__main__":
    pool = mp.Pool() # mp.cpu_count()
    it = [(arms_mean, A, K, n, f, p_ERa, p_ERf, T) for s in range(sample)]
    results = pool.starmap(run_algo_COOPAndIndep, it)
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
    # bound = UB(x, n+f, K, coop.alpha_feed, coop.alpha_agents)
    pp.figure(figsize=(10, 6))
    pp.plot(x, means, label = "COOPalgo", color='blue')
    pp.fill_between(x, means - errors,means + errors, color='blue', alpha=0.2)
    pp.plot(x, means_indepp, label = "independent", color='red')
    pp.fill_between(x, means_indepp - errors_indepp,means_indepp + errors_indepp, color='red', alpha=0.2)
    pp.grid()
    # pp.plot(x, bound, label = "theory", color='black')
    tit_g = f"n={n}, f={f}, agents={A}, arms={K}, bias0={arms_mean[0]:.3}"
    tit_g += f", q={q}, samples={sample}, ER_a={p_ERa}, ER_f={p_ERf}"
    tit_f = f"n={n}_f={f}_agents={A}_arms={K}_bias0={arms_mean[0]:.3}"
    tit_f += f"_q={q}_samples={sample}_ER_a={p_ERa}_ER_f={p_ERf}"
    pp.title(tit_g)
    pp.legend(loc = 2)
    pp.savefig(tit_f+'.pdf', dpi=600)
