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

def plot_COOPvsNOcoop(r, r_indepp, pmts, sample):
    (q, A, K, n, f, p_ERa, p_ERf, T, seed_a, seed_f, arms_mean, lr) = pmts
    means = np.mean(r, axis=0)
    errors = np.std(r, axis=0)
    means_indepp = np.mean(r_indepp, axis=0)
    errors_indepp = np.std(r_indepp, axis=0)
    x = [i for i in range(T)]
    # bound = UB(x, n+f, K, coop.alpha_feed, coop.alpha_agents)
    pp.figure(figsize=(8, 6))
    # pp.plot(x, means, label = r"EXP3-$\alpha^2$", color='blue')
    pp.plot(x, means, color='blue')
    pp.fill_between(x, means - errors,means + errors, color='blue', alpha=0.2)
    # pp.plot(x, means_indepp, label = "No Cooperation", color='red')
    pp.plot(x, means_indepp, color='red')
    pp.fill_between(x, means_indepp - errors_indepp, means_indepp + errors_indepp, color='red', alpha=0.2)
    pp.grid()
    # pp.plot(x, bound, label = "theory", color='black')
    tit_g = f"q={q}  $p_{{ER}}^N$={p_ERa}  $p_{{ER}}^F$={p_ERf}"
    tit_f = f"q={q}_n={n}_f={f}_agents={A}_arms={K}_T={T}_samples={sample}"
    tit_f += f"_bias0={arms_mean[0]:.3}_pER_a={p_ERa}_pER_f={p_ERf}"
    # pp.title(tit_g, size=16)
    # pp.xlabel("Number of Rounds")
    # pp.ylabel("Network Regret")
    # pp.legend(loc=2)
    pp.savefig('G/'+tit_f+'.pdf', dpi=600)
    return pp

def plot_GraphGraph(pmts):
    w = 6; h = 6; d = 300;
    (q, A, K, n, f, p_ERa, p_ERf, T, seed_a, seed_f) = pmts
    tit_a = f"G_a__A={A}_pER_a={p_ERa}_seed_a={seed_a}_n={n}.pdf"
    tit_g_a = f"A={A}  $p_{{ER}}^N$={p_ERa}"
    tit_f = f"G_f__K={K}_pER_f={p_ERf}_seed_f={seed_f}_f={f}.pdf"
    tit_g_f = f"K={K}  $p_{{ER}}^F$={p_ERf}"
    pp.figure(figsize=(w, h), dpi=d)
    net_feed = nx.erdos_renyi_graph(K, p_ERf, seed = seed_f)
    nx.draw_networkx(net_feed, pos=nx.spring_layout(net_feed), node_color ="orange", alpha=0.9)
    pp.axis ("off")
    # pp.title(tit_g_f, size=16)
    pp.savefig('G/'+tit_f)
    pp.figure(figsize=(w, h), dpi=d)
    net_agents = nx.erdos_renyi_graph(A, p_ERa, seed = seed_a)
    nx.draw_networkx(net_agents, pos=nx.spring_layout(net_agents), node_color ="lightgreen", alpha=0.9)
    pp.axis ("off")
    # pp.title(tit_g_a, size=16)
    pp.savefig('G/'+tit_a)

def UB(x_list, d, K, alpha_feed, alpha_agents):
    f = lambda x: d+2*np.sqrt(np.log(K)*x*(alpha_feed/(1-np.exp(-1))*(alpha_agents/coop.Q+1)+d))
    l = [x for x in map(f, x_list)]
    return l

def run_algo(q, A, K, n, f, p_ERa, p_ERf, T, seed_a, seed_f, arms_mean, lr):
    ban = bandit.Bandit(arms_mean, A, K, n, f, p_ERa, p_ERf, q, seed_a, seed_f)
    coop = bandit.COOP_algo(ban, T)
    r = np.array([])
    for t in range(T):
        coop.act(lr)
        r = np.append(r, ban.total_regret())
        ###### SOME PRINT ######################################################
        if (t+1) % 1 == 0:
            print("-----> Round:", t+1)
            print("eta[0] =", coop.eta[0], "eta[1] =", coop.eta[1], "|| T =",coop.T, "|| alpha_f =", coop.alpha_feed,
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

def run_experiment(q = [1,0.5,1/20], A = [20], K = [20], n = [2], f = [2],
                    p_ERa = [0.1], p_ERf = [0.1], seed_a = 42, seed_f = 41,
                    T = 1000, sample = 10, LB_bias = True, bias=0.,
                    lr='dt', cpu_num = None):
    if __name__ == "__main__":
        comb_pmts = [[*pmts, T, seed_a, seed_f] for pmts in itertools.product(q, A, K, n, f, p_ERa, p_ERf)]
        summary_results = []
        for pmts in comb_pmts:
            plot_GraphGraph(pmts)
            (q, A, K, n, f, p_ERa, p_ERf, T, seed_a, seed_f) = pmts
            arms_mean = 1/2 * np.ones(K)
            if LB_bias:
                arms_mean[0] = 1/2 - np.sqrt(K/T)
            else:
                arms_mean[0] = bias
            pmts.append(arms_mean)
            pmts.append(lr)
            pmts_indep = [q, A, K, 0, f, 0, p_ERf, T, seed_a, seed_f, arms_mean, lr]
            it = [pmts for s in range(sample)]
            it_indep = [pmts_indep for s in range(sample)]
            pool = mp.Pool(cpu_num) # mp.cpu_count()
            results = pool.starmap(run_algo, it+it_indep)
            pool.close(); pool.join()
            results = np.array(results)
            r = results[:sample,:]
            r_indepp = results[sample:,:]
            plot_COOPvsNOcoop(r, r_indepp, pmts, sample)
            summary_results.append((q, p_ERa, p_ERf, np.mean(r[:,-1]), np.mean(r_indepp[:,-1])))
        print(summary_results)
    return 0

# run_experiment(q=[0.05, 0.5, 1], f=[1], n=[1], K=[20], A=[20], p_ERa = [0.8, 0.2], p_ERf = [0.8, 0.2],
#                 T=1000, sample=20, LB_bias=True, bias=0., lr = 'dt', cpu_num=None,
#                 seed_a = 43, seed_f = 41)
run_experiment(q=[1], f=[1], n=[1], K=[20], A=[20], p_ERa = [0.8], p_ERf = [0.8],
                T=10, sample=3, LB_bias=False, bias=0., lr = 'dt', cpu_num=None,
                seed_a = 43, seed_f = 41)
