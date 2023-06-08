import numpy as np
import population 
class Connection:
    def __init__(self,in_node,out_node,is_enabled) -> None:
        self.in_node = in_node
        self.out_node = out_node
        self.is_enabled = is_enabled
        self.weight = self.generate_random_weight()
        self.innovation_number = self.generate_inovation_number()


    def set_is_enabled(self, is_enabled):
        self.is_enabled = is_enabled

    def set_weight(self, new_weight):
        self.weight = new_weight

    def generate_random_weight(self):
        return np.random.normal(0,0.5)

    def generate_inovation_number(self):
        for gene in population.Population.population_gens:
            if(self == gene):
                return gene.innovation_number
        
        inovation_number = population.Population._global_inovation_count 
        population.Population._global_inovation_count +=1
        population.Population.population_gens.append(self)
        return inovation_number

    def __eq__(self, other) -> bool:
        return (self.in_node == other.in_node) and (self.out_node == other.out_node)

    def __str__(self) -> str:
        return str(self.in_node) + "--->" + str(self.out_node) + " I:" + str(self.innovation_number) + "  enabled:" + str(self.is_enabled)  + " weight:" + str(self.weight)
    
    