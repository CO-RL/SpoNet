import torch
import matplotlib.pyplot as plt
import numpy as np
import random
import copy
import time



class SimulatedAnnealing:
    # Constructor:
    #   - Data_Dict: file object from 'open_file' method
    #   - Iter: int for the number of iterations annealing runs for
    #   - Alpha: Parameter for decreasing temperature
    #   - Beta: Parameter for increasing the # of inner-loop iterations
    def __init__(self, p, n, r, dist, iter, alpha=None, beta=None):
        # Get dataset variables
        self.p = p
        self.r = r
        self.num_vertices = n
        self.vertices = list(range(0, n))
        self.dist = dist
        self.best_iter = 0
        self.best_solution = None
        self.best_score = None

        # Generate initial solution randomly
        self.S = np.zeros(self.num_vertices)
        selected_vertices = random.sample(self.vertices, self.p)
        for i in range(self.num_vertices):
            if i + 1 in selected_vertices:
                self.S[i] = 1

        # Initialize SA variables
        self.T = 10
        self.iterations = iter
        self.alpha = alpha if alpha else 0.9
        self.beta = beta if beta else 1.02

    # Method to begin annealing process
    def start(self):
        count = 0
        timer = 0

        while self.T > 1:
            i = 0
            while i < self.iterations:
                # Perturb S to get new solution
                new_S = self.perturb(self.S)

                # Calculate scores for both solutions
                h_S = self.score(self.S)
                h_new_S = self.score(new_S)

                # Simulated annealing condition statement
                random_num = random.uniform(0, 1)
                if (h_new_S < h_S) or (random_num < np.exp((h_S - h_new_S) / self.T)):
                    self.S = copy.deepcopy(new_S)

                    # Set aside best score
                    if self.best_solution is None:
                        self.best_solution = copy.deepcopy(self.S)
                        self.best_score = h_S
                        self.best_iter = count + 1
                    if h_new_S < self.best_score:
                        self.best_solution = copy.deepcopy(new_S)
                        self.best_score = h_new_S
                        self.best_iter = count + 1

                # Print scores and increment counters
                i += 1
                count += 1

            # Update temp and iterations with parameters
            self.T = self.alpha * self.T
            self.iterations = self.beta * self.iterations

            # Update timer
            timer += 1
        print("第%s代最优值：%s" % (timer, self.best_score))

    # Method to perturb solution
    def perturb(self, S):
        # Randomly choose index and flip bit
        solution = copy.deepcopy(S)
        index = random.randint(0, self.num_vertices - 1)
        if solution[index] == 0:
            solution[index] = 1
        elif solution[index] == 1:
            solution[index] = 0

        # Fix up solution if infeasible
        while np.sum(solution) < self.p:
            solution[random.choice(np.where(solution == 0)[0])] = 1
        while np.sum(solution) > self.p:
            solution[random.choice(np.where(solution == 1)[0])] = 0

        # Return new solution
        return solution

    # Method to score solutions
    def score(self, solution):
        # Get list of centers and remaining points to be assigned
        centers = np.where(solution == 1)[0]
        rem_points = np.where(solution == 0)[0]

        no_cover = self.num_vertices - self.p
        for i in rem_points:
            for j in centers:
                if self.dist[i, j] > self.r:
                    continue
                else:
                    no_cover -= 1
                    break
        return no_cover


if __name__ == "__main__":
    # n = 20
    # p = 4
    # r = 0.3
    # torch.manual_seed(123)
    # points = torch.FloatTensor(n, 2).uniform_(0, 1)
    # dist = (points[None, :, :] - points[:, None, :]).norm(p=2, dim=-1)
    # t1 = time.time()
    # SA = SimulatedAnnealing(p=p, n=n, r=r, dist=dist, iter=100)
    # SA.start()
    # print(time.time()-t1)
    # best_solution = [i for i, x in enumerate(SA.best_solution) if x == 1]
    # print(best_solution)
    # fig = plt.figure(figsize=(18, 8))
    # ax1 = fig.add_subplot(1, 2, 1)
    # display_points_with_mclp(points, ax1, best_solution, r)
    # ax1.set_title(f'The result of MCLP by SA\nThe objective value: {round(SA.best_score, 4)}')
    # plt.show()


    num_sample = 10
    n = 200
    p = 15
    r = 0.15
    torch.manual_seed(1234)

    datas = []
    dists = []
    centers = []
    objs = []
    # sols = []
    for i in range(num_sample):
        data = torch.FloatTensor(n, 2).uniform_(0, 1)
        dist = (data[None, :, :] - data[:, None, :]).norm(p=2, dim=-1)
        datas.append(data)
        dists.append(dist)
    start_time = time.time()
    for i in range(num_sample):
        points = datas[i]
        dist = dists[i]
        np_dist = np.array(dist)
        SA = SimulatedAnnealing(p=p, n=n, r=r, dist=dist, iter=5)
        SA.start()
        obj = n - SA.best_score
        # best_solution = [i for i, x in enumerate(SA.best_solution) if x == 1]
        objs.append(obj)
    end_time = time.time() - start_time

    average_obj = np.mean(objs)
    print(f"The number of instances is: {num_sample}")
    print(f"The total solution time is: {end_time}")
    print(f"The average solution time is: {end_time / num_sample}")
    print(f"The average objective is: {average_obj}")
