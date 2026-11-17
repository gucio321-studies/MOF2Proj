import numpy as np
import pandas


class Data:
    def __init__(
        self, *, nodesPosFilePath="data/11.dat", nodesMapFilePath="data/19.dat"
    ):
        nodesPos = pandas.read_csv(nodesPosFilePath, sep="\\s\\s+", engine="python")
        nodesPos = nodesPos.sort_values(by="nr")  # make sure nodes are sorted
        numNodes = len(nodesPos["nr"])
        self.nodes = np.column_stack((
            nodesPos["x(nm)"],
            nodesPos["y(nm)"],
            nodesPos["indeks brzegu"],
        ))

        nodesMap = pandas.read_csv(nodesMapFilePath, sep="\\s\\s+", engine="python")
        ni = 9  # TODO: this is hardcoded but probably should not be
        self.nodesMap = np.zeros((int(len(nodesMap['nr elementu "m"']) / ni), ni))
        self.nodesMap[
            nodesMap['nr elementu "m"'].astype(int) - 1,
            nodesMap['lokalny nr węzła "i"'].astype(int) - 1,
        ] = (
            nodesMap['globalny numer węzła "nlg(m,i)"'] - 1
        )

    def V(self, vmax):
        result = np.full(self.nodes.shape[0], None)
        bottom = np.where(self.nodes[:,2]==2)
        top = np.where(self.nodes[:,2]==3)
        side = np.where(self.nodes[:,2]==1)
        sideY = np.sort(np.unique(self.nodes[side,1]))
        result[bottom] = 0
        result[top] = vmax

        yMin = np.min(self.nodes[side, 1])
        yMax = np.max(self.nodes[side, 1])
        result[side] = vmax * (self.nodes[side,1] - yMin)/(-yMin+yMax)
        return result

