from collections import namedtuple
import sys
import torch
import torch.optim as optim

from tqdm import tqdm  # training progress bar
from itertools import count

# Import local classes
sys.path.append(".")
import Constants
from NNnetwork import NNnetwork
from ReplayMemory import ReplayMemory
from TicTacToe_env import TicTacToe
from Action_Selection import select_action
from Optimize_model import optimize_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


policy_net = NNnetwork().to(device)
# The target_net will be used to evaluate the value function of state/action tuples,
# and will be updated from the policy_net occasionally
target_net = NNnetwork().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters())
memory = ReplayMemory(Constants.MEMORY_SIZE)

print('\nTraining Started')

steps_done = 0

for i_episode in tqdm(range(Constants.num_episodes)):
    # Initialize the environment and state
    game = TicTacToe()
    state = game.board
    for t in count():  # infinite loop until game is done
        # Select and perform an action
        action = select_action(policy_net, state, steps_done)
        done, reward = game.step(action)
        reward = torch.tensor([reward], device=device)

        # Observe new state
        if not done:
            next_state = game.board
            game.negate_board()  # to utilize both of the players to learn
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization
        optimize_model(policy_net, target_net, optimizer, memory)
        if done:
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % Constants.TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('\nTraining Completed')

print("\nexample of a game")

game = TicTacToe()
state = game.board
done = 0
while done == 0:
    policy_net.eval()
    action = torch.argmax(policy_net.forward(game.board))
    print("action:", action)
    done, reward = game.step(action=action)
    print("done:", done, "reward:", reward)
    print("board after first move:")
    print(game.board.reshape((3, 3)))
    if done == 0:
        game.negate_board()
        policy_net.eval()
        action = torch.argmax(policy_net.forward(game.board))
        print("action:", action)
        done, reward = game.step(action=action)
        print("done:", done, "reward:", reward)
        game.negate_board()
        print("board after move:")
        print(game.board.reshape((3, 3)))
