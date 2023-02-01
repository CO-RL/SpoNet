from torch.utils.data import Dataset
import torch
import os
import pickle
from problems.LSCP.state_LSCP import StateLSCP
# from problems.LSCP.state_LSCP_cover import StateLSCP

class LSCP(object):
    NAME = 'LSCP'

    @staticmethod
    def get_facility_num(dataset, pi):
        loc = dataset['loc']
        # radius = dataset['radius'][0]
        # theta = dataset['theta'][0]
        batch_size, n_loc, _ = loc.size()
        facilities_num = []
        for idx, sol in enumerate(pi):
            solution = sol[sol > -1]
            facility_num = len(solution)
            facilities_num.append(facility_num)

        # facilities_num = torch.cat(facilities_num)
        facilities_num = torch.as_tensor(facilities_num, dtype=torch.float32)
        return facilities_num

    @staticmethod
    def make_dataset(*args, **kwargs):
        return LCSPDataset(*args, **kwargs)

    @staticmethod
    def make_state(*args, **kwargs):
        return StateLSCP.initialize(*args, **kwargs)


class LCSPDataset(Dataset):
    def __init__(self, filename=None, size=50, num_samples=5000, offset=0, p=None, r=0.15, theta=1, distribution=None):
        super(LCSPDataset, self).__init__()

        if filename is not None:
            assert os.path.splitext(filename)[1] == '.pkl'

            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.data = [torch.FloatTensor(row) for row in (data[offset:offset + num_samples])]
        else:
            # Sample points randomly in [0, 1] square
            self.data = [dict(loc=torch.FloatTensor(size, 2).uniform_(0, 1), radius=r, theta=theta)
                         for i in range(num_samples)]

        self.size = len(self.data)
        self.radius = r
        self.theta = theta

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return self.data[idx]
