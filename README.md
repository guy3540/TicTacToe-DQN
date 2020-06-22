# Find the optimal strategy to play Tic Tac Toe using DQN.
This is a beginner level assignment, and is based on pytorch's DQN tutorial.
It uses two similar networks:
  1. Policy network which learns the optimal policy to maximize the reward
  2. Target network which estimates the value of taking each action in a certain state

The project also implements Experience Replay to decorrelate batches of state trasitions in the training process.

# To run
In order to train an agent, simply run main.py. 

You may want to edit the Constants.py file to change any constant, for example the number of epochs in the training session.

Also- the agent's model is defined in NNnetwork.py. As a default, the model will consist 3 Fully Connected layers, where the first two FC layers are followed by batch normalization and ReLU layers.

Once the file is run, the agent will begin to train using a decaying-epsilon-greedy policy (choose a random action in a decreasing rate along the training session).

Once the training session is done, a game will be played between two instances of the finalized version of the agent.
