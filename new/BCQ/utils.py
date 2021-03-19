import numpy as np
import torch
import setting

# Generic replay buffer for standard gym tasks
class StandardBuffer(object):
    def __init__(self, state_dim, batch_size, buffer_size, device):
        self.batch_size = setting.BATCH_SIZE
        self.max_size = int(buffer_size)
        self.device = device

#         self.ptr = 0
        self.crt_size = 0

        self.state = np.zeros((self.max_size, state_dim))
        self.action = np.zeros((self.max_size, 1))
        self.next_state = np.array(self.state)
        self.reward = np.zeros((self.max_size, 1))
        self.not_done = np.zeros((self.max_size, 1))
        self.crt_size = self.max_size

    def add(self, state, action, next_state, reward, done):
        self.state = state
        self.action = action
        self.next_state = next_state
        self.reward = reward
        self.not_done = 1. - done
        # self.ptr = (self.ptr + 1) % self.max_size



    def sample(self):
        ind = np.random.randint(0, self.crt_size, size=self.batch_size)
        return (
            torch.FloatTensor(self.state[ind]).to(self.device),
            torch.LongTensor(self.action[ind]).to(self.device),
            torch.FloatTensor(self.next_state[ind]).to(self.device),
            torch.FloatTensor(self.reward[ind]).to(self.device),
            torch.FloatTensor(self.not_done[ind]).to(self.device)
        )


