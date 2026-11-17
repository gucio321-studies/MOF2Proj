from matplotlib import pyplot as plt
from . import data, mes

class Task():
    def __init__(self):
        self.data = data.Data()
        self.mapper = mes.Mapper(data=self.data)

    def __call__(self):
        self.task2()

    def task1(self):
        plt.plot(self.data.nodes[:,0], self.data.nodes[:,1], marker='o', linestyle='')
        plt.show()
    def task2(self):
        v=self.data.V(5)
        plt.scatter(self.data.nodes[:,0], self.data.nodes[:,1], c=v)
        plt.show()
