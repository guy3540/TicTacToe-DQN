import numpy as np
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TicTacToe:
    def __init__(self):
        self.board = None
        self.active_player = None
        self.reset_board()

    def reset_board(self):
        self.board = torch.tensor(np.zeros((1, 9)), device=device).float()
        self.active_player = 1

    def step(self, action):
        done, reward = self.take_action(action)
        return done, reward

    def take_action(self, action):
        # input and action legal status check
        assert np.all([action <= 8, action >= 0]), "coordinate invalid"
        if self.board[0, action] != 0:
            # Illegal move, reset and return negative reward
            self.reset_board()
            done, reward = 1, torch.tensor([-1], device=device).float()
            # print(reward)
            return done, reward
        # update board and return reward
        self.board[0, action] = torch.tensor([self.active_player]).float()
        self.active_player = -self.active_player
        # if game is over, reset board
        if self.check_winner() != 0:
            # self.reset_board()
            done, reward = 1, torch.tensor([1], device=device).float()
            # print(reward)
        elif self.board.nonzero().nelement() >= 18:  # board is filled and no winner
            done, reward = 1, torch.tensor([0], device=device).float()
        else:
            done, reward = 0, torch.tensor([0], device=device).float()
            # print(reward)
        return done, reward

    def negate_board(self):
        self.board = -1 * self.board
        self.active_player = -self.active_player

    def check_winner(self):
        board = self.board.reshape((3, 3))
        for i in range(board.shape[0]):
            if np.all([board[i, 0] == board[i, 1] == board[i, 2],
                       board[i, 0] != 0]):
                return board[i, 0]

        for j in range(board.shape[0]):
            if np.all([board[0, j] == board[1, j] == board[2, j],
                       board[0, j] != 0]):
                return board[0, j]

        if np.all([board[0, 0] == board[1, 1] == board[2, 2],
                   board[0, 0] != 0]):
            return board[0, 0]

        if np.all([board[2, 0] == board[1, 1] == board[0, 2],
                   board[2, 0] != 0]):
            return board[2, 0]

        return 0