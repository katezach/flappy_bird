

from utils import RELU,SIGMOID,TANH

class Config:
    
    def __init__(self,
                 no_inputs,
                 no_outputs,
                 

                 lamda = 300,

                 stagnetion_limit = 50,

                 miu_prop = 0.3,


                 crossover_rate = 0.8,
                 crossover_rate_of_reenable = 0.2,

                 activation_funtions = [RELU, SIGMOID, TANH],
                 activation_funtion_mutation_rate = 0.2,
                 

                 weight_mutation_rate = 0.9,
                 step_size = 1,
                 remove_node_mutation_rate = 0.5,
                 add_node_mutation_rate = 0.3,
                 remove_connection_mutation_rate = 0.5,
                 add_connection_muttation_rate = 0.3,
                 target_species=30,

                 C1 = 1,
                 C2 = 1,
                 C3 = 0.5,
                 delta = 5,
                 ) -> None:
        self.no_inputs = no_inputs
        self.no_outputs = no_outputs
        

        self.lamda = lamda

        self.stagnetion_limit = stagnetion_limit

        self.target_species = target_species
        self.miu_prop = miu_prop


        self.crossover_rate = crossover_rate
        self.crossover_rate_of_reenable = crossover_rate_of_reenable

        self.activation_funtions = activation_funtions
        self.activation_funtion_mutation_rate = activation_funtion_mutation_rate
        self.weight_mutation_rate = weight_mutation_rate
        self.step_size = step_size
        self.add_node_mutation_rate = add_node_mutation_rate
        self.remove_node_mutation_rate = remove_node_mutation_rate
        self.add_connection_mutation_rate = add_connection_muttation_rate
        self.remove_connection_mutation_rate = remove_connection_mutation_rate


        self.C1 = C1
        self.C2 = C2
        self.C3 = C3
        self.delta = delta