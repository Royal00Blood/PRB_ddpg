import numpy as np
from collections import namedtuple

# Определение named tuple для хранения переходов
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.ones((capacity,), dtype=np.float32)
        self.position = 0

    def push(self, *args):
        """Сохранение перехода в буфер"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = Transition(*args)
        self.priorities[np.clip(self.position, 0, len(self.priorities) - 1)] = max(self.priorities.max(), 1) # Инициализация приоритета максимальным значением
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Выборка batch_size переходов с вероятностью, пропорциональной их приоритетам"""
        total = len(self.buffer)
        if total < batch_size:
            return None, None, None
        if total == 0:
            return [], [], []
        if total != len(self.priorities):
            self.priorities = np.zeros((total,), dtype=np.float32)
        probabilities = (self.priorities+ 1e-5) ** self.alpha
        probabilities /= probabilities.sum()
        indices = np.random.choice(total, batch_size, p=probabilities)
        transitions = [self.buffer[idx] for idx in indices]
        weights = (total * probabilities[indices]) ** (-0.5)
        weights /= weights.max()
        return transitions, indices, weights
        
    def update_priority(self, index, priority):
        #if 0 <= index < self.capacity:
        # self.priorities[index] = priority
        for i, p in zip(index, priority):
            self.priorities[i] = p

    def __len__(self):
        return len(self.buffer)
    
    def __getitem__(self, idx):
        return self.memory[idx]

