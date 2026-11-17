import numpy as np


class Mapper:
    def __init__(self, *, data):
        self.data = data

    def x(self, *, k, ksi):
        """
        k is a TRANSLATED node number
        """
        return np.sum(
            [
                self.data.nodes[self.data.nodesMap[k, i].astype(int)][0]
                * Mapper.gi(i=i, ksi=ksi)
                for i in range(4)
            ]
        )

    def x(self, *, k, ksi):
        """
        k is a TRANSLATED node number
        """
        return np.sum(
            [
                self.data.nodes[self.data.nodesMap[k, i].astype(int)][1]
                * Mapper.gi(i=i, ksi=ksi)
                for i in range(4)
            ]
        )

    @staticmethod
    def gi(*, i, ksi):
        if i == 0:
            return Mapper.g1(ksi)
        elif i == 1:
            return Mapper.g2(ksi)
        elif i == 2:
            return Mapper.g3(ksi)
        elif i == 3:
            return Mapper.g4(ksi)

    @staticmethod
    def g1(epsilon):
        return Mapper.f1(epsilon[0]) * Mapper.f1(epsilon[1])

    @staticmethod
    def g2(epsilon):
        return Mapper.f2(epsilon[0]) * Mapper.f1(epsilon[1])

    @staticmethod
    def g3(epsilon):
        return Mapper.f1(epsilon[0]) * Mapper.f2(epsilon[1])

    @staticmethod
    def g4(epsilon):
        return Mapper.f2(epsilon[0]) * Mapper.f2(epsilon[1])

    @staticmethod
    def f1(epsilon):
        return (1 - epsilon) / 2

    @staticmethod
    def f2(epsilon):
        return (1 + epsilon) / 2
