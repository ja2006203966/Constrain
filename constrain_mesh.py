
import sys
sys.setrecursionlimit(3000)
import time
import numpy as np
from tqdm import tqdm


class Inteval():
    def __init__(self, index, arr):
        self.index = np.array(index)
        self.i = index[0]
        self.j = index[1]
        self.idx_str = self.encode(index)
        # self.N = None
        # self.meshGrid = None
        self.value = arr[tuple(index)]
        self.lines = []
        self.poles = []
        self.ans = []
    def encode(self, index):
        return '_'.join([str(i) for i in index])
    def decode(self, idx_str):
        return [int(i) for i in idx_str.split('_')]
    def get_lines(self,):
        if len(self.lines) != 0:
            return self.lines
        self.poles = [list(self.index), [self.i+1, self.j], [self.i+1, self.j+1], [self.i, self.j+1]]
        self.lines = [[self.poles[i-1], pt] for i, pt in enumerate(self.poles)]
        return self.lines
    # def get_grid(self, N = 10):
    #     if N == self.N:
    #         return self.meshGrid
    #     x = np.linspace(self.x, self.x + self.l, N)
    #     y = np.linspace(self.y, self.y + self.l, N)
    #     xv, yv = np.meshgrid(x, y)
    #     xv = np.expand_dims(xv, axis = -1)
    #     yv = np.expand_dims(yv, axis = -1)
    #     self.N = N
    #     return np.concatenate([xv, yv], axis = -1)



def length(v):
    return np.sqrt(np.sum(v**2, axis = -1))
def _dp_line_constrain(pts, fn, bias = 1e-3):
    pts = np.array(pts)
    condis = np.sign(fn(pts).flatten())
    if abs(np.sum(condis)) == 2:
        return None
    if np.any(condis == 0):
        if condis[1] == 0:
            return pts[1]
        return pts[0]
    pt = np.mean(pts, axis = 0)
    c = np.sign(fn(pt))
    if (c == 0) or (length(pts[1] - pts[0]) < bias):
        return pt
    if c == condis[0]:
        return _dp_line_constrain([pt, pts[1]], fn, bias = bias)
    if c == condis[1]:
        return _dp_line_constrain([pts[0], pt], fn, bias = bias)
    print(f"Unexpect value {pt, pts, c, condis}")
    raise f"Unexpect value {pt, pts, c, condis}"

if __name__ == '__main__':
    a = np.zeros((100,100))
    a[1, 1] = 0.5
    a[0, :] += 1
    a[:, 0] += 1
    a[0, 0] = 1
    import matplotlib.pyplot as plt
    
    search_table = dict()
    for i in tqdm(range(a.shape[0])):
        for j in range(a.shape[1]):
            itv = Inteval([i, j], a)
            search_table[itv.idx_str] = itv
    
    def fn(a):
        """a: array like with shape (2, 2)
        """
        return np.sum(a**2, axis = -1) - (80)**2
    for k in tqdm(search_table):
        for l in (itv:=search_table[k]).get_lines():
            pt = _dp_line_constrain(l, fn, bias = 1e-3)
            # try:
                # pt = _dp_line_constrain(l, fn, bias = 1e-8)
            # except:
                # print(k)
            if type(pt) == type(None):
                continue
            itv.ans.append(pt)
    fig, ax = plt.subplots()
    ax.imshow(a, origin = 'lower')
    for k in search_table:
        itv = search_table[k]
        if len(itv.ans) == 0:
            continue
        # print(k, itv.ans)
        lines = np.array(itv.ans)
        # print(lines)
        ax.plot(lines[:, 0], lines[:, 1], c = 'white')

    plt.show()