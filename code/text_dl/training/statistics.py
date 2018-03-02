import warnings

class Statistics():
    def __init__(self, capacity):
        self.storage = [{}]
        self.capacity = capacity

    def update_stat(self, name, value):
        if not len(self.storage) > self.capacity:
            idx = len(self.storage) - 1
            self.storage[idx][name] = value
    
    def step(self):
        if len(self.storage) == self.capacity + 1:
            warnings.warn("The number of steps have already reached capacity, not adding new stats")
        self.storage.append({})

    def get_stat(self, idx, name):
        '''
        Raises:
            - `IndexError` exception if idx out of range
            - `KeyError` exception if statistics name does not exist
        '''
        return self.storage[idx][name]

    