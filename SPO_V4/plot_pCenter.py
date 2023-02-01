import torch
import pickle
import numpy as np
from algorithm.pCenter.SA import SimulatedAnnealing
import matplotlib.pyplot as plt
from plot import display_points_with_pc, display_points_with_p_center
from algorithm.pCenter.gurobi import gurobi_solver_p_center

if __name__ == "__main__":
    ## load data
    filename = "../data/PC/PC_100.pkl"
    with open(filename, 'rb') as f:
        data = pickle.load(f)

    seed = np.random.randint(0, len(data))
    instance = data[seed]
    loc = instance['loc']
    p = instance['p']
    n = len(loc)
    dist = (loc[None, :, :] - loc[:, None, :]).norm(p=2, dim=-1)
    np_dist = np.array(dist)

    ##  Gurobi solver
    x, y, gurobi_obj = gurobi_solver_p_center(loc, np_dist, p)

    ## SpoNet
    filename = "./results/PC/PC_100/PC_100-PC100_epoch-499-sample1280-t1-0-10000.pkl"
    f = open(filename, 'rb')
    f = pickle.load(f)
    spo_result = f[0][seed]
    spo_obj = spo_result[0]
    spo_centers = torch.tensor(spo_result[1])

    ## SA alg
    SA = SimulatedAnnealing(p=p, n=n, m=n, dist=dist, iter=500)
    SA.start()
    SA_obj = SA.best_score
    best_solution = [i for i, x in enumerate(SA.best_solution) if x == 1]
    best_solution = torch.tensor(best_solution)

    ## plot
    print('Start plotting...')
    fig = plt.figure(figsize=(20, 6))
    print("Plot the solution of p-Center:")
    ax1 = fig.add_subplot(1, 3, 1)
    display_points_with_pc(loc, gurobi_obj, y)
    ax1.set_title(f"The result of p-Center by Gurobi\nThe objective distance: {round(gurobi_obj,4)}")

    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_title(f"The result of p-Center by SpoNet\nThe objective distance: {round(float(spo_obj),4)}")
    display_points_with_p_center(loc, spo_obj, spo_centers)

    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_title(f"The result of p-Center by SA\nThe objective distance: {round(float(SA_obj),4)}")
    display_points_with_p_center(loc, SA_obj, best_solution)
    plt.show()
