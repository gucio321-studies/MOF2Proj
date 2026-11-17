from matplotlib import pyplot as plt
from . import data, mes

class Task():
    def __init__(self):
        self.data = data.Data()
        self.mapper = mes.Mapper(data=self.data)
        print(self.mapper.x(k=49, ksi=[.25, -.25]))

    def __call__(self):
        self.task1()

    def task1(self):
        plt.plot(self.data.nodes[:,0], self.data.nodes[:,1], marker='o', linestyle='')
        plt.show()
