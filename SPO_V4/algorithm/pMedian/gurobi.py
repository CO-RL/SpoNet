from gurobipy import *
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
# from plot import display_points_with_pmedian

def gurobi_solver_p_median(loc, C, P):
    # Problem data
    # C = (loc[:, None, :] - loc[None, :, :]).norm(p=2, dim=-1)
    N = loc.size()[0]
    # C = np.array(C)

    model = Model('p-Median')
    model.setParam('OutputFlag', False)
    model.setParam('MIPFocus', 2)
    model.setParam(GRB.Param.TimeLimit, 600.0)
    # Add variables
    x = {}
    y = {}

    # Add Client Decision Variables and Service Decision Variables
    for i in range(N):
        y[i] = model.addVar(vtype="B", name="y(%s)" % i)
        for j in range(N):
            x[i, j] = model.addVar(vtype="B", name="x(%s, %s)" % (i, j))
    # Update Model Variables
    model.update()
    #     Set Objective Function
    model.setObjective(quicksum(C[i, j] * x[i, j] for i in range(N) for j in range(N)))
    #     Add Constraints
    model.addConstr(quicksum(y[i] for i in range(N)) == P)
    for j in range(N):
        model.addConstr(quicksum(x[i, j] for i in range(N)) == 1)
    # for i in range(N):
    #     for j in range(N):
    #         model.addConstr()
    model.addConstrs(x[i, j] <= y[i] for j in range(N) for i in range(N))

    model.optimize()

    # return a stardard result list

    x_result = np.zeros((N, N))
    y_result = np.zeros(N)
    for i in range(N):
        y_result[i] = y[i].X
        for j in range(N):
            x_result[i, j] = x[i, j].X
    obj = model.ObjVal
    return x_result, y_result, obj


if __name__ == '__main__':

    num_sample = 10
    n = 1000
    p = 15
    torch.manual_seed(1234)
    datas = []
    dists = []
    centers = []
    objs = []
    for i in range(num_sample):
        data = torch.FloatTensor(n, 2).uniform_(0, 1)
        dist = (data[None, :, :] - data[:, None, :]).norm(p=2, dim=-1)
        datas.append(data)
        dists.append(dist)
    start_time = time.time()
    for i in range(num_sample):
        print(i)
        points = datas[i]
        dist = dists[i]
        np_dist = np.array(dist)
        x, y, obj = gurobi_solver_p_median(points, np_dist, p)
        objs.append(obj)
        # centers.append(result[1])
    end_time = time.time() - start_time
    average_obj = np.mean(objs)
    print(f"The number of instances is: {num_sample}")
    print(f"The total solution time is: {end_time}")
    print(f"The average solution time is: {end_time/num_sample}")
    print(f"The average objective is: {average_obj}")

    # n_points = 20
    # p = 4
    # data = torch.FloatTensor(n_points, 2).uniform_(0, 1).squeeze(-1)
    #
    # start = time.time()
    # x, y, obj = gurobi_solver_p_median(data, p)
    # solution_time = time.time() - start

    # ax, fig = plt.figure(figsize=(16, 8))
    # display_points_with_pmedian(data, x, y)
    # ax.set_title(f'The result of P-median by Gurobi\nThe objective distance: {round(obj, 4)}')