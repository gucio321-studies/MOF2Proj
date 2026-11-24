import numpy as np


class Mapper:
    def __init__(self, *, data, delta=1e-5):
        self.data = data
        self.delta = delta
        self.__p = np.array(
            [
                -np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)),
                np.sqrt(3 / 7 - 2 / 7 * np.sqrt(6 / 5)),
                np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5)),
                -np.sqrt(3 / 7 + 2 / 7 * np.sqrt(6 / 5)),
            ]
        )
        self.__h = np.array(
            [
                self.h1,
                self.h2,
                self.h3,
                self.h4,
                self.h5,
                self.h6,
                self.h7,
                self.h8,
                self.h9,
            ]
        )

    def x(self, *, k, ksi):
        """
        k is a TRANSLATED node number
        """
        return np.sum(
            [
                self.data.nodes[self.data.nodesMap[k, i].astype(int), 0]
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
                self.data.nodes[self.data.nodesMap[k, i].astype(int), 1]
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

    @staticmethod
    def q1(ksi):
        return ksi * (ksi - 1) / 2

    @staticmethod
    def q2(ksi):
        return (1 - ksi) * (ksi + 1)

    @staticmethod
    def q3(ksi):
        return (ksi) * (ksi + 1) / 2

    @staticmethod
    def h1(ksi1, ksi2):
        return Mapper.q1(ksi1) * Mapper.q1(ksi2)

    @staticmethod
    def h2(ksi1, ksi2):
        return Mapper.q3(ksi1) * Mapper.q1(ksi2)

    @staticmethod
    def h3(ksi1, ksi2):
        return Mapper.q1(ksi1) * Mapper.q3(ksi2)

    @staticmethod
    def h4(ksi1, ksi2):
        return Mapper.q3(ksi1) * Mapper.q3(ksi2)

    @staticmethod
    def h5(ksi1, ksi2):
        return Mapper.q2(ksi1) * Mapper.q1(ksi2)

    @staticmethod
    def h6(ksi1, ksi2):
        return Mapper.q3(ksi1) * Mapper.q2(ksi2)

    @staticmethod
    def h7(ksi1, ksi2):
        return Mapper.q1(ksi1) * Mapper.q2(ksi2)

    @staticmethod
    def h8(ksi1, ksi2):
        return Mapper.q2(ksi1) * Mapper.q3(ksi2)

    @staticmethod
    def h9(ksi1, ksi2):
        return Mapper.q2(ksi1) * Mapper.q2(ksi2)

    def h(self, i, ksi1, ksi2):
        return self.__h[i](ksi1, ksi2)

    @staticmethod
    def w(n):
        if n in [0, 1]:
            return (18 + np.sqrt(30)) / 36
        elif n in [2, 3]:
            return (18 - np.sqrt(30)) / 36
        else:
            raise ValueError("n out of range")

    def p(self, n):
        return self.__p[n]

    def dhidksi2(self, *, j, l, n):
        return (
            self.h(i=j, ksi1=self.p(l), ksi2=(self.p(n) + self.delta))
            - self.h(i=j, ksi1=self.p(l), ksi2=(self.p(n) - self.delta))
        ) / (2 * self.delta)

    def dhidksi1(self, *, j, l, n):
        return (
            self.h(i=j, ksi1=self.p(l) + self.delta, ksi2=(self.p(n)))
            - self.h(i=j, ksi1=self.p(l) - self.delta, ksi2=(self.p(n)))
        ) / (2 * self.delta)

    def tij(self, i, j):
        return np.sum(
            [
                np.sum(
                    [
                        self.w(l)
                        * self.w(n)
                        * (
                            self.dhidksi1(j=j, l=l, n=n) * self.dhidksi1(j=i, l=l, n=n)
                            + self.dhidksi2(j=j, l=l, n=n)
                            * self.dhidksi2(j=i, l=l, n=n)
                        )
                        for n in range(4)
                    ]
                )
                for l in range(4)
            ]
        )

    def t(self):
        return np.array([[self.tij(i=i, j=j) for i in range(9)] for j in range(9)])

    def T(self):
        result = np.zeros((self.data.N, self.data.N))
        for i in range(self.data.M):
            for i1 in range(9):
                for i2 in range(9):
                    result[
                        self.data.nodesMap[i, i1].astype(int),
                        self.data.nodesMap[i, i2].astype(int),
                    ] += self.tij(i=i1, j=i2)

        return result
    def T_with_BC(self):
        T = self.T().copy()
        boundary = np.where(self.data.nodes[:,2]!=0)
        T[boundary, :] = 0
        T[boundary, boundary] = 1
        return T

    def solve(self):
        p = self.data.V(vmax=100e-3)
        p[p==None] = 0
        p = p.astype(float)
        T = self.T_with_BC()
        d = np.linalg.solve(T, p)
        return d
