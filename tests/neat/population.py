from config import Config
from genome import Genome
import numpy as np
from species import Species
from utils import BIAS_TYPE,INPUT_TYPE,OUTPUT_TYPE,HIDDEN_TYPE

def f(network):
    input = []
    for i in range(13):
        input.append(i)

    x = network.predict(input)
    return np.sum(x)


class Population:
    _global_inovation_count = 0
    population_gens = []
    _global_node_count = 0
    popultion_nodes = []
   
    def __init__(self, config , fitness_function = f) -> None:
        self.species_count = 0
        self.config = config
        self.population = self._generate_initial_population()
        self.fitness_function = fitness_function
        self.species = []
        self.best = self.population[0]
        self.best_fitness = -10

    def train(self,no_epochs, observer ):
        for epoch in range(no_epochs):
            print(f"\nEpoch: {epoch}")
            # for genome in self.population:
            #     print(genome)
            for genome in self.population:
                genome.set_fitness(self.fitness_function(genome.create_network()))

            self.population = sorted( self.population, key=lambda x: x.fitness, reverse=True)
        
            if(self.population[0].fitness > self.best_fitness):
                self.best_fitness = self.population[0].fitness
                self.best = self.population[0]
            print(f"Best {self.best_fitness} ")
            print(f"Generation best {self.population[0].fitness}")
            fitnesses = [genome.fitness for genome in self.population]
            average_fitness = np.mean(fitnesses)
            std_fitness = np.std(fitnesses)
            print(f"Generation average {average_fitness}, stdev {std_fitness}")
            # print(self.__str__())

            self._specilise()
            
            observer.add_fitness(self.population[0].fitness)
            observer.add_species(self.species)

            print(f"Species {len(self.species)}")
            if(len(self.species) > self.config.target_species):
                self.config.delta *= 1.2
            if(len(self.species) < self.config.target_species):
                self.config.delta *= 0.9
            self._compute_size_species()
            self._reduce_species()
            self._reproduce()


    def _specilise(self):
        for genome in self.population:
            added = False
            for sp in self.species:
                if(sp.distance_to_specie(genome) <= self.config.delta):
                    sp.add_genome(genome)
                    added = True
                    break
            if(not added):
                self.species_count += 1
                id = self.species_count
                # print(id)
                sp = Species(genome,id)
                self.species.append(sp)
        self.species[:] = [sp for sp in self.species if len(sp.genomes) > 0]
        for sp in self.species:
            sp.update_species()

   

    def _compute_size_species(self):
        sum_fitnesses = max(0.00001, np.sum([sp.average_fitness for sp in self.species]))
        for sp in self.species:
            sp.childrens_required = int(self.config.lamda * sp.average_fitness/ sum_fitnesses)
            # print(sp.childrens_required)
                                        
    def _reduce_species(self):
        new_species = []
        for sp in self.species: 
            if(sp.stagnation > self.config.stagnetion_limit or sp.childrens_required == 0):
                continue
            sp.genomes = sp.genomes[:max(1,int(self.config.miu_prop*len(sp.genomes)))]
            new_species.append(sp)
        self.species = new_species

    def _reproduce(self):
        new_population = []
        for sp in  self.species:
            self.population.append(sp.genomes[0])
            count = 1
            while(count < sp.childrens_required):
                parent1 = sp.tournament_selection()

                if(np.random.random() < self.config.crossover_rate):
                    #2parents reproduction
                    parent2 = sp.tournament_selection()
                    

                    if(parent1.fitness > parent2.fitness):
                        children = Genome.crossover(parent1,parent2)
                    elif(parent1.fitness > parent2.fitness):
                        children = Genome.crossover(parent2,parent1)
                    else:
                        if(len(parent1.connection_genes) >= len(parent2.connection_genes)):
                            children = Genome.crossover(parent1,parent2)
                        else:
                            children = Genome.crossover(parent2,parent1)
                else:
                    #asexual reproduction
                    children = Genome.crossover(parent1,parent1)

                children.mutation()

                new_population.append(children)
                count +=1
            sp.end_of_epoch()

        print(f"No of basic genomes: {self.config.lamda - len(new_population)}" )
        while(len(new_population) < self.config.lamda):

            parent = self.best
           

            children = Genome.crossover(parent,parent)
            children.mutation()
            new_population.append(children)

        self.population = new_population

    def _create_basic_genome(self):
        genome = Genome(self.config)
        inputs = []
        outputs = []
        
        new_node = genome.add_node_gene(0,BIAS_TYPE)
        inputs.append(new_node)

        for i in range(self.config.no_inputs):
            new_node = genome.add_node_gene(i+1,INPUT_TYPE)
            inputs.append(new_node)


        for i in range(self.config.no_outputs):
            new_node = genome.add_node_gene(self.config.no_inputs + i + 1, OUTPUT_TYPE)
            outputs.append(new_node)
        
        for output in outputs:
            genome.add_connection_gene(0,output.id)
        

        for input in inputs:
            for output in outputs:
                genome.add_connection_gene(input.id,output.id)
        
        return genome

    def _generate_initial_population(self):
        population = []

        for _ in range(self.config.lamda):
            
            genome = self._create_basic_genome()
            population.append(genome)

        return population

    pass

    def __str__(self) -> str:
        for genome in self.population:
            print(genome)