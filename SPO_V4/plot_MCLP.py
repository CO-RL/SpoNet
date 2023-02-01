import torch
import pickle
import numpy as np
from algorithm.MCLP.SA import SimulatedAnnealing
import matplotlib.pyplot as plt
from plot import display_points_with_mclp
from algorithm.MCLP.gurobi import gurobi_solver_MCLP


if __name__ == "__main__":

    ## load data
    filename = "../data/MCLP/MCLP_100.pkl"
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    seed = np.random.randint(0, len(data))
    instance = data[seed]
    loc = instance['loc']
    p = instance['p']
    r = instance['r']
    n = len(loc)
    dist = (loc[None, :, :] - loc[:, None, :]).norm(p=2, dim=-1)
    np_dist = np.array(dist)

    ##  Gurobi solver
    x, y, gurobi_obj = gurobi_solver_MCLP(loc, p, np_dist, r)
    solution1 = [id for id, key in enumerate(x) if key == 1]

    ## SpoNet
    filename = "./results/MCLP/MCLP_100/MCLP_100-MCLP100_epoch-499-sample1280-t1-0-10000.pkl"
    f = open(filename, 'rb')
    f = pickle.load(f)
    spo_result = f[0][seed]
    spo_obj = -spo_result[0]
    spo_centers = torch.tensor(spo_result[1])

    ## SA alg
    SA = SimulatedAnnealing(p=p, n=n, r=r, dist=dist, iter=500)
    SA.start()
    SA_obj = n - SA.best_score
    best_solution = [i for i, x in enumerate(SA.best_solution) if x == 1]


    # fig = plt.figure(figsize=(18, 8))
    # ax1 = fig.add_subplot(1, 2, 1)
    # display_points_with_mclp(points, ax1, best_solution, r)
    # ax1.set_title(f'The result of MCLP by SA\nThe objective value: {round(SA.best_score, 4)}')
    # plt.show()

    ## plot
    print('Start plotting...')
    fig = plt.figure(figsize=(20, 6))
    print("Plot the solution of p-Median:")
    ax1 = fig.add_subplot(1, 3, 1)
    display_points_with_mclp(loc, ax1, solution1, r)
    ax1.set_title(f"The result of MCLP by Gurobi\nThe objective distance: {gurobi_obj}")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title(f"The result of MCLP by SpoNet\nThe objective distance: {spo_obj}")
    display_points_with_mclp(loc, ax2, spo_centers, r)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title(f"The result of MCLP by SA\nThe objective distance: {SA_obj}")
    display_points_with_mclp(loc, ax3, best_solution, r)
    plt.show()
