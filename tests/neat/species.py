from genome import Genome
import numpy as np
class Species:
    def __init__(self,main_genome, id = 0) -> None:
        self.id = id
        self.main_genome = main_genome
        self.genomes = [main_genome]

        self.stagnation = 0

        self.childrens_required = 0

        self.max_fitness = main_genome.fitness
        self.average_fitness = 0.0



    def tournament_selection(self):
        fitnesses = np.array([genome.fitness - self.genomes[-1].fitness + 0.01 for genome in self.genomes])
        prob = fitnesses/np.sum(fitnesses)
        loc = np.random.choice(len(fitnesses), p=prob)
        return self.genomes[loc]

    def add_genome(self,genome):
        self.genomes.append(genome)

    def end_of_epoch(self):
        self.stagnation = 0
        self.genomes = []

    def update_species(self):
        self.average_fitness = 0
        self.genomes = sorted(self.genomes,key= lambda x: x.fitness,reverse=True)

        for genome in self.genomes:
            self.average_fitness += genome.fitness
            if(genome.fitness >self.max_fitness): 
                self.main_genome = genome
                self.stagnation = 0
                self.max_fitness = genome.fitness
        self.average_fitness /= len(self.genomes)

    def distance_to_specie(self, genome):
        return Genome.distance_genomes(self.main_genome,genome)
    
    def __str__(self) -> str:
        out = ""
        for el in self.genomes:
            out += str(el.fitness) + " "
        return out