import torch
from typing import NamedTuple
from utils.boolmask import mask_long2bool, mask_long_scatter


def construct_solutions(actions):
    return actions


class StateMCLP(NamedTuple):
    # Fixed input
    loc: torch.Tensor
    p: torch.Tensor
    radius: torch.Tensor
    dist: torch.Tensor

    # If this state contains multiple copies (i.e. beam search) for the same instance, then for memory efficiency
    # the loc and dist tensors are not kept multiple times, so we need to use the ids to   index the correct rows.
    ids: torch.Tensor  # Keeps track of original fixed data index of rows

    # State
    first_a: torch.Tensor
    prev_a: torch.Tensor
    visited_: torch.Tensor  # Keeps track of nodes that have been visited
    facility: torch.Tensor  # B x p
    # mask_cover
    mask_cover: torch.Tensor
    dynamic: torch.Tensor
    dynamic_updation: torch.Tensor
    cover_num: torch.Tensor
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
        p = data['p'][0]
        radius = data['radius'][0]
        batch_size, n_loc, _ = loc.size()
        dist = (loc[:, :, None, :] - loc[:, None, :, :]).norm(p=2, dim=-1)

        facility_list = [[] for i in range(batch_size)]
        facility = torch.tensor(facility_list, device=loc.device)
        cover_num = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)
        prev_a = torch.zeros(batch_size, 1, dtype=torch.long, device=loc.device)

        return StateMCLP(
            loc=loc,
            p=p,
            radius=radius,
            dist=dist,
            ids=torch.arange(batch_size, dtype=torch.int64, device=loc.device)[:, None],  # Add steps dimension
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
            dynamic=torch.ones(batch_size, 1, n_loc, dtype=torch.float, device=loc.device),
            dynamic_updation=torch.arange(radius, device=loc.device).float().expand(batch_size, 1, -1) / radius,
            facility=facility,
            cover_num=cover_num,
            first_a=prev_a,
            prev_a=prev_a,
            i=torch.zeros(1, dtype=torch.int64, device=loc.device)  # Vector with length num_steps
        )

    def get_final_cost(self):

        assert self.all_finished()

        return self.cover_num

    def get_cover_num(self, facility):
        """
        :param facility: list, a list of facility index list,  if None, generate randomly
        :return: obj val of given facility_list
        """

        batch_size, n_loc, _ = self.loc.size()
        _, p = facility.size()
        radius = self.radius
        facility_tensor = facility.unsqueeze(-1).expand_as(torch.Tensor(batch_size, p, n_loc))
        f_u_dist_tensor = self.dist.gather(1, facility_tensor)
        mask = f_u_dist_tensor < radius
        mask = torch.sum(mask, dim=1)
        cover_num = torch.count_nonzero(mask, dim=-1)

        return cover_num

    def update(self, selected):

        # Update the state
        prev_a = selected[:, None]  # Add dimension for step
        cur_coord = self.loc[self.ids, prev_a]
        facility_list = self.facility.tolist()
        slected_list = selected.tolist()
        [facility_list[i].append(slected_list[i]) for i in range(len(selected))]

        new_facility = torch.tensor(facility_list, device=self.loc.device)
        new_cover_num = self.get_cover_num(new_facility)

        first_a = prev_a if self.i.item() == 0 else self.first_a

        if self.visited_.dtype != torch.bool:
            visited_ = mask_long_scatter(self.visited_, prev_a)
        else:
            # Add one dimension since we write a single value
            visited_ = self.visited_.scatter(-1, prev_a[:, :, None], 1)

        # mask covered cities
        batch_size, sequence_size, _ = self.loc.size()
        batch_size = self.ids.size(0)
        dists = (self.loc[self.ids.squeeze(-1)] - cur_coord).norm(p=2, dim=-1)
        mask_cover = self.mask_cover.clone()

        for i in range(batch_size):
            if len(self.radius.size()) == 0:
                n_idx = dists[i].argsort()[torch.sort(dists[i])[0] < self.radius]
            else:
                n_idx = dists[i].argsort()[torch.sort(dists[i])[0] < self.radius.squeeze(0)[selected[i]]]

            mask_cover[i, 0, n_idx] = 1

        return self._replace(first_a=first_a, visited_=visited_, mask_cover=mask_cover, prev_a=prev_a,
                             facility=new_facility, cover_num=new_cover_num, i=self.i + 1)

    def all_finished(self):
        # Exactly n steps
        return self.i == self.p

    def get_current_node(self):
        return self.prev_a

    def get_mask(self):
        return self.visited_
