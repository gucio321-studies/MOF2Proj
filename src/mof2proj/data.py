import numpy as np
import pandas


class Data:
    def __init__(
        self, *, nodesPosFilePath="data/11.dat", nodesMapFilePath="data/19.dat"
    ):
        nodesPos = pandas.read_csv(nodesPosFilePath, sep="\\s\\s+", engine="python")
        nodesPos = nodesPos.sort_values(by="nr")  # make sure nodes are sorted
        numNodes = len(nodesPos["nr"])
        self.nodes = np.array(
            list(zip(nodesPos["x(nm)"], nodesPos["y(nm)"], nodesPos["indeks brzegu"])),
            dtype=object,
        )

        nodesMap = pandas.read_csv(nodesMapFilePath, sep="\\s\\s+", engine="python")
        ni = 9  # TODO: this is hardcoded but probably should not be
        self.nodesMap = np.zeros((int(len(nodesMap['nr elementu "m"']) / ni), ni))
        self.nodesMap[
            nodesMap['nr elementu "m"'].astype(int) - 1,
            nodesMap['lokalny nr węzła "i"'].astype(int) - 1,
        ] = nodesMap['globalny numer węzła "nlg(m,i)"'] - 1
