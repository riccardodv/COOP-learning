import numpy as np
import bandit
from collections import deque
import networkx as nx
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as pp
import multiprocessing as mp
import itertools
import copy

def UB(x_list, d, K, alpha_feed, alpha_agents):
    f = lambda x: d+2*np.sqrt(np.log(K)*x*(alpha_feed/(1-np.exp(-1))*(alpha_agents/coop.Q+1)+d))
    l = [x for x in map(f, x_list)]
    return l

def run_algo(q, A, K, n, f, T, p_ERa, p_ERf, arms_mean):
    ban = bandit.Bandit(arms_mean, A, K, n, f, p_ERa, p_ERf, q)
    coop = bandit.COOP_algo(ban, T)
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
    return r



# # Draw networks
# w = 6; h = 6; d = 300;
# pp.figure(figsize=(w, h), dpi=d)
# net_feed = nx.erdos_renyi_graph(K, p_ERf, seed = seed_f)
# nx.draw_networkx(net_feed, pos=nx.spring_layout(net_feed), node_color ="red", alpha=0.9)
# pp.axis ("off")
# net_agents = nx.erdos_renyi_graph(A, p_ERa, seed = seed_a)
# nx.draw_networkx(net_agents,  pos=nx.spring_layout(net_agents), node_color ="blue", alpha=0.9)
# pp.axis ("off")
# pp.show()
# pp.savefig("out.pdf")


def run_experiment(q = [1], A = [10], K = [10], n = [2], f = [2], T = 1000, p_ERa = 0.1, p_ERf = 0.1, sample = 10):
    seed_f =41; seed_a=43
    comb_pmts = [[*pmts, T, p_ERa, p_ERf] for pmts in itertools.product(q, A, K, n, f)]

    for pmts in comb_pmts:
        if __name__ == "__main__":
            (q_, A_, K_, n_, f_, T_, p_ERa_, p_ERf_) = pmts
            arms_mean = 1/2 * np.ones(K_)
            arms_mean[0] = 1/2 - np.sqrt(K_/T_)
            pmts.append(arms_mean)
            pmts_indep = [q_, A_, K_, 0, f_, T_, 0, p_ERf_, arms_mean]
            it = [pmts for s in range(sample)]
            it_indep = [pmts_indep for s in range(sample)]
            pool = mp.Pool() # mp.cpu_count()
            results = pool.starmap(run_algo, it+it_indep)
            pool.close(); pool.join()

            results = np.array(results)
            r = results[:sample,:]
            r_indepp = results[sample:,:]
            means = np.mean(r, axis=0)
            errors = np.std(r, axis=0)
            means_indepp = np.mean(r_indepp, axis=0)
            errors_indepp = np.std(r_indepp, axis=0)
            x = [i for i in range(T)]
            # bound = UB(x, n+f, K, coop.alpha_feed, coop.alpha_agents)
            pp.figure(figsize=(10, 6))
            pp.plot(x, means, label = r"EXP3-$\alpha^2$", color='blue')
            pp.fill_between(x, means - errors,means + errors, color='blue', alpha=0.2)
            pp.plot(x, means_indepp, label = "No Cooperation", color='red')
            pp.fill_between(x, means_indepp - errors_indepp,means_indepp + errors_indepp, color='red', alpha=0.2)
            pp.grid()
            # pp.plot(x, bound, label = "theory", color='black')
            tit_g = f"n={n_}, f={f_}, agents={A_}, arms={K_}, bias0={arms_mean[0]:.3}"
            tit_g += f", q={q_}, samples={sample}, ER_a={p_ERa_}, ER_f={p_ERf_}"
            tit_f = f"q={q_}_n={n_}_f={f_}_agents={A_}_arms={K_}_T={T_}_samples={sample}"
            tit_f += f"_bias0={arms_mean[0]:.3}_ER_a={p_ERa_}_ER_f={p_ERf_}"
            pp.title(tit_g)
            pp.xlabel("Number of Rounds")
            pp.ylabel("Network Regret")
            pp.legend(loc = 2)
            pp.savefig('G/'+tit_f+'.pdf', dpi=600)
    return 0

run_experiment(q=[1, 0.5, 1/20], f=[2], n=[2], K=[20], A=[20], T=1000, sample=10)
