import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


class StateLSCP(NamedTuple):
    # Fixed input
    loc: torch.Tensor
    radius: torch.Tensor
    # theta: torch.Tensor
    dist: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    facility: torch.Tensor
    facility_num: torch.Tensor
    # mask_cover
    mask_cover: torch.Tensor
    i: torch.Tensor  # Keeps track of step

    @property
    def visited(self):
        if self.visited_.dtype == torch.bool:
            return self.visited_
        else:
            return mask_long2bool(self.visited_, n=self.loc.size(-2))

    @staticmethod
    def initialize(data, visited_dtype=torch.bool):
        loc = data['loc']
        # theta = data['theta']
        radius = data['radius'][0]
        batch_size, n_loc, _ = loc.size()
        dist = (loc[:, :, None, :] - loc[:, None, :, :]).norm(p=2, dim=-1)

        facility_list = [[] for i in range(batch_size)]
        facility = torch.tensor(facility_list, device=loc.device)
        facility_num = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)

        return StateLSCP(
            loc=loc,
            # theta=theta,
            radius=radius,
            dist=dist,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension

            prev_a=prev_a,
            facility=facility,
            facility_num=facility_num,
            # Keep visited with depot so we can scatter efficiently (if there is an action for depot)
            visited_=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, 1, n_loc,
                    dtype=torch.bool, device=loc.device
                )
                if visited_dtype == torch.bool
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            mask_cover=(  # Visited as mask is easier to understand, as long more memory efficient
                torch.zeros(
                    batch_size, 1, n_loc,
                    dtype=torch.bool, device=loc.device
                )
                if visited_dtype == torch.bool
                else torch.zeros(batch_size, 1, (n_loc + 63) // 64, dtype=torch.int64, device=loc.device)  # Ceil
            ),
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_final_cost(self):

        assert self.all_finished()

        return self.facility

    def get_facilitit_num(self, facility):
        _, p = facility.size()
        radius = self.radius
        return p

    def update(self, selected):

        # Update the state
        prev_a = selected[:, None]  # Add dimension for step
        cur_coord = self.loc[self.ids, prev_a]
        facility_list = self.facility.tolist()
        slected_list = selected.tolist()
        [facility_list[i].append(slected_list[i]) for i in range(len(selected))]

        new_facility = torch.tensor(facility_list, device=self.loc.device)
        new_facility_num = self.get_facilitit_num(new_facility)

        if self.visited_.dtype == torch.bool:
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)
        else:
            visited_ = mask_long_scatter(self.visited_, prev_a)

        # mask covered cities
        batch_size, sequence_size, _ = self.loc.size()
        batch_size = self.ids.size(0)
        dists = (self.loc[self.ids.squeeze(-1)]-cur_coord).norm(p=2, dim=-1)

        mask_cover = self.mask_cover.clone()

        for i in range(batch_size):
            if len(self.radius.size()) == 0:
                n_idx = dists[i].argsort()[torch.sort(dists[i])[0] < self.radius]
            else:
                n_idx = dists[i].argsort()[torch.sort(dists[i])[0] < self.radius.squeeze(0)[selected[i]]]

            mask_cover[i, 0, n_idx] = 1

        return self._replace(visited_=visited_, mask_cover=mask_cover, prev_a=prev_a,
                             facility=new_facility, facility_num=new_facility_num,  i=self.i + 1)

    def all_finished(self):
        # Exactly n steps
        # return (self.cover_num >=self.theta * self.mask_cover.size(-1)).all()
        # return (self.mask_cover.sum(-1) >= self.theta * self.mask_cover.size(-1)).all()
        return self.mask_cover.all()

    def get_finished(self):
        # return (self.cover_num >= self.theta * self.mask_cover.size(-1))
        return self.mask_cover.sum(-1) == self.mask_cover.size(-1)

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        return self.visited

