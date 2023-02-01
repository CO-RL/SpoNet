import torch
import pickle
import numpy as np
from algorithm.pMedian.simulated_annealing import simulated_annealing
import matplotlib.pyplot as plt
from plot import display_points_with_pm, display_points_with_pmedian
from algorithm.pMedian.gurobi import gurobi_solver_p_median

if __name__ == "__main__":

    ## load data
    filename = "../data/PM/PM_100.pkl"
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    seed = np.random.randint(0, len(data))
    instance = data[seed]
    loc = instance['loc']
    p = instance['p']
    dist = (loc[None, :, :] - loc[:, None, :]).norm(p=2, dim=-1)
    np_dist = np.array(dist)

    ##  Gurobi solver
    x, y, gurobi_obj = gurobi_solver_p_median(loc, dist, p)

    ## SpoNet
    filename = "./results/PM/PM_100_15/PM_100_15-PM100_epoch-499-sample1280-t1-0-10000.pkl"
    f = open(filename, 'rb')
    f = pickle.load(f)
    spo_result = f[0][seed]
    spo_obj = spo_result[0]
    # spobj = round(spo_obj.astype(np.float64), 4)
    spo_centers = torch.tensor(spo_result[1])

    ## SA alg
    re = simulated_annealing(np_dist, p, verbose=True)
    result = re[1]
    SA_centers = torch.tensor(result[1], dtype=torch.int32)
    SA_obj = result[0]

    ## plot
    print('Start plotting...')
    fig = plt.figure(figsize=(20, 6))
    print("Plot the solution of p-Median:")
    ax1 = fig.add_subplot(1, 3, 1)
    display_points_with_pmedian(loc, x, y)
    ax1.set_title(f"The result of p-Median by Gurobi\nThe objective distance: {round(gurobi_obj, 4)}")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title(f"The result of p-Median by SpoNet\nThe objective distance: {round(float(spo_obj), 4)}")
    # ax2.set_title(f"The result of p-Median by SpoNet\nThe objective distance: {round(float(obj), 4)}")
    display_points_with_pm(loc, spo_centers)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title(f"The result of p-Median by SA\nThe objective distance: {round(SA_obj,4)}")
    display_points_with_pm(loc, SA_centers)
    plt.show()
