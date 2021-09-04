import numpy as np

x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
choice = np.random.choice(range(len(x)), size=(5,), replace=False)
ind = np.zeros(len(x), dtype=bool)
ind[choice] = True
rest = ~ind
a = x[ind]
b = x[rest]
print(a)
print(b)
