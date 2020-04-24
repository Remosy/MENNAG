import history
import numpy as np

def Q_divide(B, g, init):
    Bg = B.copy()
    if (not init):
        for i in range(len(g)):
            Bg[i, i] -= sum(B[i, :])
    w, v = np.linalg.eigh(Bg)
    signs = v[:,np.argmax(w)]
    g1 = []
    g2 = []
    s = np.zeros(len(g))
    for i in range(len(g)):
        if (signs[i] > 0):
            g1.append(g[i])
            s[i] = 1
        else:
            g2.append(g[i])
            s[i] = -1
    Q = s@Bg@s
    if (Q <= 0 or g1 == [] or g2 == []):
        return 0, [g]
    Q1, groups1 = Q_divide(Bg[g1][:,g1], list(range(len(g1))), False)
    Q2, groups2 = Q_divide(Bg[g2][:,g2], list(range(len(g2))), False)
    groups = []
    for group in groups1:
        tmp = []
        for x in group:
            tmp.append(g1[x])
        groups.append(tmp)
    for group in groups2:
        tmp = []
        for x in group:
            tmp.append(g2[x])
        groups.append(tmp)
    return Q + Q1 + Q2, groups

def compute_B(nn):
    n = nn.nodeCount
    m = nn.connList.connCount
    B = np.zeros((n, n))
    k = np.zeros(n)
    for i in range(m):
        source = nn.connList.connSource[i]
        target = nn.connList.connTarget[i]
        B[source, target] += 1
        B[target, source] += 1
        k[source] += 1
        k[target] += 1
    for i in range(n):
        for j in range(n):
            B[i, j] -= k[i] * k[j] / (2 * m)
    return B

def Q(nn):
    nn.compile()
    n = nn.nodeCount
    m = nn.connList.connCount
    if (m < 1):
        return 0, []
    B = compute_B(nn)
    Q, groups = Q_divide(B, list(range(n)), True)
    Q = Q / (4 * m)
    return (Q, groups)


class Analyzer:

    def __init__(self, hist):
        self.hist = hist

    def Q_metric(self, index):
        nn = self.hist.bests[index].execute()
        nn.compile()
        n = nn.nodeCount
        m = nn.connList.connCount
        B = compute_B(nn)
        Q, groups = Q_divide(B, list(range(n)), True)
        Q = Q / (4 * m)
        return Q
