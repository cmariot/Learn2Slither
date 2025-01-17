class Directions:

    def __init__(self):
        self.directions = {
            'up': (0, -1),
            'down': (0, 1),
            'left': (-1, 0),
            'right': (1, 0)
        }
        self._iter_index = 0
        self._keys = list(self.directions.keys())

    def get_directions(self):
        return self.directions

    def get_direction(self, direction):
        return self.directions[direction]

    def keys(self):
        return self.directions.keys()

    def get_key_by_index(self, index):
        if index < 0 or index >= len(self._keys):
            raise IndexError("Index out of range in Directions")
        return self._keys[index]

    def items(self):
        return self.directions.items()

    def values(self):
        return self.directions.values()

    def value_list(self):
        return list(self.directions.values())

    def __iter__(self):
        self._iter_index = 0
        return self

    def __next__(self):
        if self._iter_index < len(self._keys):
            key = self._keys[self._iter_index]
            self._iter_index += 1
            return key, self.directions[key]
        else:
            self._iter_index = 0
            raise StopIteration
