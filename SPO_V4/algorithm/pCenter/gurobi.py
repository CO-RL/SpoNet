from gurobipy import *
import numpy as np

def gurobi_solver_p_center(loc, C, P):
    # Problem data
    # C = (loc[:, None, :] - loc[None, :, :]).norm(p=2, dim=-1)
    N = loc.size()[0]
    # C = np.array(C)

    model = Model('p-Center')
    model.setParam('OutputFlag', False)
    model.setParam('MIPFocus', 2)
    model.setParam(GRB.Param.TimeLimit, 600.0)
    # Add variables
    x = {}
    y = {}
    # Add Client Decision Variables and Service Decision Variables
    z = model.addVar(name = "z")
    for i in range(N):
        y[i] = model.addVar(vtype="B", name="y(%s)" % i)
        for j in range(N):
            x[i, j] = model.addVar(vtype="B", name="x(%s, %s)" % (i, j))
    # Update Model Variables
    model.update()
    #     Set Objective Function
    model.setObjective(z)
    # model.setObjective(quicksum(C[i, j] * x[i, j] for i in range(N) for j in range(N)))
    #     Add Constraints
    model.addConstr(quicksum(y[i] for i in range(N)) == P)
    for j in range(N):
        model.addConstr(quicksum(x[i, j] for i in range(N)) == 1)

    model.addConstrs(x[i, j] <= y[i] for j in range(N) for i in range(N))
    model.addConstrs(quicksum(C[i, j] * x[i, j] for i in range(N)) <= z for j in range(N))
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

