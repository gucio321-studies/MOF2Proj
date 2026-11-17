import numpy as np
import pandas

class Data:
    def __init__(self, nodesPosFilePath="data/11.dat"):
        nodesPos = pandas.read_csv(nodesPosFilePath, sep="\\s\\s+", engine='python')
        nodesPos = nodesPos.sort_values(by='nr') # make sure nodes are sorted
        numNodes = len(nodesPos['nr'])
        self.nodes = np.array(
            list(zip(nodesPos['x(nm)'], nodesPos['y(nm)'], nodesPos['indeks brzegu'])),
            dtype=object
        )
