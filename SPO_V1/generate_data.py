import argparse
import os
import numpy as np
import torch

from utils.data_utils import check_extension, save_dataset


def generate_PM_data(n_samples, n_users, n_facilities, p):
    data = [dict(users=torch.FloatTensor(n_users, 2).uniform_(0, 1),
                 facilities=torch.FloatTensor(n_facilities, 2).uniform_(0, 1),
                 p=p,
                 ) for i in range(n_samples)]
    return data


def generate_PC_data(n_samples, n_users, n_facilities, p):
    data = [dict(users=torch.FloatTensor(n_users, 2).uniform_(0, 1),
                 facilities=torch.FloatTensor(n_facilities, 2).uniform_(0, 1),
                 p=p,
                 ) for i in range(n_samples)]
    return data


def generate_MCLP_data(n_samples, n_users, n_facilities, p, radius):
    data = [dict(users=torch.FloatTensor(n_users, 2).uniform_(0, 1),
                 facilities=torch.FloatTensor(n_facilities, 2).uniform_(0, 1),
                 p=p,
                 radius=radius,
                 ) for i in range(n_samples)]
    return data


def generate_LSCP_data(n_samples, n_users, n_facilities, radius):
    data = [dict(users=torch.FloatTensor(n_users, 2).uniform_(0, 1),
                 facilities=torch.FloatTensor(n_facilities, 2).uniform_(0, 1),
                 radius=radius,
                 ) for i in range(n_samples)]
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", help="Filename of the dataset to create (ignores datadir)")
    parser.add_argument("--data_dir", default='data', help="Create datasets in data_dir/problem (default 'data')")
    parser.add_argument("--problem", type=str, default='PM',
                        help="Problem, 'PM', 'PC', 'MCLP' or 'LSCP' to generate")

    parser.add_argument("--dataset_size", type=int, default=10000, help="Size of the dataset")
    parser.add_argument('--n_users', type=int, nargs='+', default=50,
                        help="number of users")
    parser.add_argument('--n_facilities', type=int, nargs='+', default=20,
                        help="number of facilities")
    parser.add_argument('--p', type=int, nargs='+', default=4,
                        help="number of centers")
    parser.add_argument("-f", action='store_true', help="Set true to overwrite")
    parser.add_argument('--seed', type=int, default=1234, help="Random seed")

    opts = parser.parse_args()

    assert opts.filename is None or (len(opts.problems) == 1 and len(opts.graph_sizes) == 1), \
        "Can only specify filename when generating a single dataset"

    problem = opts.problem
    n_users = opts.n_users
    n_facilities = opts.n_facilities
    p = 4

    datadir = os.path.join(opts.data_dir, problem)
    os.makedirs(datadir, exist_ok=True)

    if problem == 'PM':
        filename = os.path.join(datadir, f"{problem}{n_users}_{n_facilities}_{p}.pkl")
        dataset = generate_PM_data(opts.dataset_size, n_users, n_facilities, p)
    elif problem == 'PC':
        filename = os.path.join(datadir, f"{problem}{n_users}_{n_facilities}_{p}.pkl")
        dataset = generate_PC_data(opts.dataset_size, n_users, n_facilities, p)
    elif problem == 'MCLP':
        filename = os.path.join(datadir, f"{problem}{n_users}_{n_facilities}_{p}.pkl")
        dataset = generate_PC_data(opts.dataset_size, n_users, n_facilities, p)
    elif problem == 'LSCP':
        radius = 0.1
        filename = os.path.join(datadir, f"{problem}{n_users}{n_facilities}_{radius}.pkl")
        dataset = generate_PC_data(opts.dataset_size, n_users, n_facilities, radius)
    else:
        assert False, "Unknown problem: {}".format(problem)

    save_dataset(dataset, filename)



