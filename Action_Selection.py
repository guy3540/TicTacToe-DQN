import random
import math
import torch
import Constants
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def select_action(policy_net, state, steps_done):
    sample = random.random()
    eps_threshold = Constants.EPS_END + (Constants.EPS_START - Constants.EPS_END) * \
        math.exp(-1. * steps_done / Constants.EPS_DECAY)
    steps_done += 1
    train_flag = policy_net.training
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            policy_net.eval()
            action = policy_net(state).max(1)[1].view(1, 1)
    else:
        action = torch.tensor([[random.randrange(Constants.n_actions)]], device=device, dtype=torch.long)
    if not train_flag:
        print(state.reshape((3, 3)))
    else:
        policy_net.train()
    return action


def select_random_action(state):
    with torch.no_grad():
        valid_idx = (state == 0).nonzero()
        choice = torch.multinomial(torch.arange(valid_idx.size(0)).float(), 1)
        # print("choice", choice)
        return choice.clone().detach()


# def select_target_action(policy_net, target_net, state):
#     with torch.no_grad():
#         train_flag = policy_net.training
#         target_net.eval()
#         action = target_net(state).max(1)[1].view(1, 1)
#         if not train_flag:
#             print("Board after policy_net move:")
#             print(state)
#         else:
#             target_net.train()
#         return action
