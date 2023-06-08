import matplotlib.pyplot as plt
import numpy as np

class Observer:
    def __init__(self) -> None:
        self.fitnesses = []
        self.species = []
        pass


    def add_fitness(self,fitness):
        self.fitnesses.append(fitness)

    def add_species(self, sps):
        
        species = {}
        for sp in sps:
            # print(sp.id)
            species[sp.id] = len(sp.genomes)
        
        self.species.append(species)


    def create_species_plot(self, no_species):  
        n = len(self.species)
        values = {}
        for i in range(1,no_species+1):
            x1 = np.zeros(n) +301
            x2 = np.zeros(n) +301
            values[i] = (x1,x2)

        # print(values)

        
        for pos in range(n):
            x = [val for val in self.species[pos].keys()]
            x.sort()
            # print(x)
            count = 0 
            for id in x:
                values[id][0][pos] = count
                values[id][1][pos] = count +  self.species[pos][id]
                count = count +  self.species[pos][id]
        # print(values)
        figure = plt.figure(figsize=(10,8))
        iterations = [i+1 for i in range(n)]
        for x in values.keys():
            maxi = n
            for pos in range(1,n):
                if(values[x][0][pos] == 301 and values[x][0][pos-1] <301 ):
                    maxi = pos
                    if(maxi<n):
                        used = x
                        for y in range(x,0,-1):
                            if(values[y][1][maxi] != 301):
                                values[x][0][maxi] = values[y][1][maxi]
                                values[x][1][maxi] = values[y][1][maxi]
                                used = y
                                break
                        if(used == x):
                            values[x][0][maxi] = 0
                            values[x][1][maxi] = 0

                    break
            plt.plot(iterations[:maxi+1],values[x][0][:maxi+1],color="grey")
            plt.plot(iterations[:maxi+1],values[x][1][:maxi+1],color="grey")
            plt.fill_between(iterations[:maxi+1],values[x][0][:maxi+1], values[x][1][:maxi+1],alpha=.5)

        plt.xlabel("Epoch")
        plt.ylabel("Size")
        plt.ylim([0,300])
        plt.xlim([1,n])
        # plt.xticks(iterations)
        return figure


  