from neuralnet import NeuralNet, Unit
from node import Node
from connection import Connection
import numpy as np
import population 
from utils import HIDDEN_TYPE,OUTPUT_TYPE,INPUT_TYPE,BIAS_TYPE,NAMES
from copy import deepcopy

class Genome:
   
    def __init__(self,config) -> None:
        self.config = config
        self.nodes_genes = []
        self.connection_genes = []
        self.fitness = None

    def set_fitness(self,fitness):
        self.fitness = fitness


    def get_gene_by_inovation_number(self, innovation_number):

        for conn in self.connection_genes:
            if(conn.innovation_number == innovation_number):
                return conn

        return None


    def mutation(self):
       
        for con in self.connection_genes:
            if(np.random.random() < self.config.weight_mutation_rate):
                con.weight += np.random.normal(0,self.config.step_size)

        for node in self.nodes_genes:
            if(np.random.random() < self.config.activation_funtion_mutation_rate):
                node.activation_function = self.config.activation_funtions[np.random.randint(len(self.config.activation_funtions))]
                
        if(np.random.random() < self.config.remove_node_mutation_rate):
            self.remove_node_mutation()
        
        if(np.random.random() < self.config.add_node_mutation_rate):
            self.add_node_mutation()

        if(np.random.random() < self.config.remove_connection_mutation_rate):
            self.remove_connection_mutation()

        if(np.random.random() < self.config.add_connection_mutation_rate):
            self.add_connection_mutation()

        

    def add_node_mutation(self):

        connection = self._get_random_connection()
        if(connection is not None):
            connection.set_is_enabled(False)

            node_in = connection.in_node
            node_out = connection.out_node

            node_id = connection.innovation_number + self.config.no_inputs + self.config.no_outputs +1
            node = self.add_node_gene(node_id,HIDDEN_TYPE)

            self.add_connection_gene(node_in,node.id,is_enabled=True,weight=1)
            self.add_connection_gene(node.id,node_out,is_enabled=True,weight=connection.weight)
        

    
    def add_connection_mutation(self):
        inputs = [node.id for node in self.nodes_genes if node.type != OUTPUT_TYPE]

        outputs = [node.id for node in self.nodes_genes if node.type != INPUT_TYPE and node.type != BIAS_TYPE]


        node_in = np.random.choice(inputs)
        node_out = np.random.choice(outputs)

        if(not self._existing_connection(node_in,node_out) and not self._create_circle(node_in, node_out)):
            self.add_connection_gene(node_in,node_out)
       
            

    def remove_connection_mutation(self):
        if(len(self.connection_genes) > 1):
            pos = np.random.randint(len(self.connection_genes))
            
            del self.connection_genes[pos]
        
            

    def remove_node_mutation(self):
        # print(self)
        node_positions = [i for i in range(len(self.nodes_genes)) if self.nodes_genes[i].type == HIDDEN_TYPE]
        if(len(node_positions) > 0):
            pos = np.random.choice(node_positions)
            
            id_node = self.nodes_genes[pos].id
            # print(id_node)
            connections_to_delete = []
            for i  in range(len(self.connection_genes)):
                if(self.connection_genes[i].in_node == id_node or self.connection_genes[i].out_node == id_node ):
                    connections_to_delete.insert(0,i)
                    # print(self.connection_genes[i])
            del self.nodes_genes[pos]

            for pos in connections_to_delete:
                del self.connection_genes[pos]

        # print(self)
        
       



    def add_node_gene(self,node_id,node_type):
        activation_function = self.config.activation_funtions[np.random.randint(len(self.config.activation_funtions))]
        node = Node(node_id,node_type,activation_function)
        self.nodes_genes.append(node)
        return node

    def add_connection_gene(self,in_node,output_node,is_enabled=True, weight=None):
        connection = Connection(in_node,output_node,is_enabled)

        if(weight is not None):
            connection.set_weight(weight)

        self.connection_genes.append(connection)
        

    def  _get_random_connection(self):
        if(len(self.connection_genes) > 0):
            return self.connection_genes[np.random.randint(0,len(self.connection_genes))]
        return None


    def  _existing_connection(self, node_in, node_out):
        for connection in self.connection_genes:
            if(connection.in_node == node_in and connection.out_node == node_out):
                return True
        
        return False

    def _create_circle(self,node_in, node_out):
        if(node_in == node_out):
            return True
        visited = [node_out]
        queue = [node_out]
        while(len(queue) > 0):
            node = queue.pop(0)
            for el in self.connection_genes:
                if(el.in_node == node):
                    if(el.out_node == node_in):
                        return True
                    if(el.out_node not in visited):
                        visited.append(el.out_node)
                        queue.append(el.out_node)
        return False

    def _add_copy_connection(self,connection):
        self.connection_genes.append(connection)

    def _add_copy_node(self,node):
        self.nodes_genes.append(node)


    def create_network(self):
        inputs_bias = [node for node in self.nodes_genes if node.type == INPUT_TYPE or node.type == BIAS_TYPE]
        units = {}
        inputs = []
        for node in inputs_bias:
            unit = Unit(node.id, [], [], activation_function=node.activation_function)
            units[node.id] = unit
            inputs.append(unit)

        outputs_ids = [node.id for node in self.nodes_genes if node.type == OUTPUT_TYPE]
        outputs = []

       


        layers = []

        visited = [node.id for node in self.nodes_genes if node.type == INPUT_TYPE or node.type == BIAS_TYPE]
        connections = [con for con in self.connection_genes if con.is_enabled]
        
        out_nodes = np.unique([con.out_node for con in connections])
        # print(out_nodes)
        # print("ASDDAS ", end=" ")
        # for el in self.nodes_genes:
        #     print(el.id,end=" ")
        # print()
        ok = False
        
        for node in self.nodes_genes:
            if(node.type != INPUT_TYPE and node.type != BIAS_TYPE and node.id not in out_nodes):
                # print(node.id,end = " ")
                unit = Unit(node.id, [], [], activation_function= node.activation_function)
                units[node.id] = unit
                visited.append(node.id)
                if(node.type == OUTPUT_TYPE):
                    outputs.append(unit)
                    # print(unit)
        # if(ok):
        #     # print(self)
        #     print() 

        while(True):
            new_layer = np.unique([con.out_node for con in connections if con.in_node in visited and con.out_node not in visited])
            layer = []
            aux = []
            for node in new_layer:
                if(any(con for con in connections if con.in_node not in visited and con.out_node == node)):
                    continue
                if node not in aux:
                    previous = []
                    weights = []
                    for el in connections:
                        if (el.out_node == node):
                            previous.append(units[el.in_node])
                            weights.append(el.weight)
                
                    unit = Unit(node,previous,weights)
                    
                    if(node in outputs_ids):
                        outputs.append(unit)
                    units[node] = unit
                    layer.append(unit)
                    aux.append(node)
            
            if(len(layer) == 0):
                break
            
            layers.append(layer)
            
            visited = np.union1d(visited , aux)
        
        inputs = sorted(inputs, key=lambda x: x.id)
        for node in [node for node in self.nodes_genes if node.type == OUTPUT_TYPE and node.id not in visited]:
            unit = Unit(node.id, [], [], activation_function=node.activation_function)
            outputs.append(unit)
        outputs = sorted(outputs, key=lambda x: x.id)
        
        if(len(outputs)<2):
            print(visited)
            print(self)
            for el in inputs:
                print(el, end=" ")
            print()
            for l in layers:
                for el in l:
                    print(el, end=" ")
                print()
            
        return NeuralNet(inputs,outputs,layers)


    def compare_all(self,other):
        
        biggest_inov_number_other = np.max([el.innovation_number for el in other.connection_genes])
        excess = len([el.innovation_number for el in self.connection_genes if el.innovation_number > biggest_inov_number_other])

        common = 0
        diff = 0

        for con in other.connection_genes:
            gene = self.get_gene_by_inovation_number(con.innovation_number)
            if(gene is not None):
                common +=1
                diff += gene.weight - con.weight

        if common == 0.0:
            common = 1.0
        average =  diff / common

        
        disjoint = len(self.connection_genes) + len(other.connection_genes) - excess - 2 * common

        return disjoint,excess,average

    def __str__(self) -> str:
        list_of_nodes = ""
        for node in self.nodes_genes:
            list_of_nodes+=str(node) + "\n"
        list_of_connections = ""
        for connection in self.connection_genes:
            list_of_connections+=str(connection) + "\n"
        return "fit: "+str(self.fitness) + " \n " + list_of_nodes + list_of_connections 
    

    @staticmethod
    def crossover(parent1,parent2):
        #parent1 is the better parent -> better fitness or equal fitness but shorter
        children = Genome(parent1.config)

        biggest_inovation = -1
        nodes_ids = []
        for parent1_gene in parent1.connection_genes:
            
            biggest_inovation = max(biggest_inovation,parent1_gene.innovation_number)
            parent2_gene = parent2.get_gene_by_inovation_number(parent1_gene.innovation_number)

            if(parent2_gene is not None):
                
                if(np.random.random()<0.5):
                    children_gene = deepcopy(parent1_gene)
                else:
                    children_gene = deepcopy(parent2_gene)
                
                children_gene.weight = (parent1_gene.weight + parent2_gene.weight)/2 
            else:
                children_gene = deepcopy(parent1_gene)

            if not children_gene.is_enabled:

                if np.random.random()<parent1.config.crossover_rate_of_reenable or parent1_gene.is_enabled:
                    children_gene.is_enabled = True


            nodes_ids.append(children_gene.in_node)
            nodes_ids.append(children_gene.out_node)
            children._add_copy_connection(children_gene)


        # for parent2_gene in parent2.connection_genes:
        #     if(parent2_gene.innovation_number < biggest_inovation and parent1.get_gene_by_inovation_number(parent2_gene.innovation_number) is None):
        #         children_gene = deepcopy(parent2_gene)

        #         nodes_ids.append(children_gene.in_node)
        #         nodes_ids.append(children_gene.out_node)
        #         children._add_copy_connection(children_gene)

        used = []

        for parent1_gene in parent1.nodes_genes:
            if((parent1_gene.id in nodes_ids) and (parent1_gene.id not in used)):
                used.append(parent1_gene.id)
                children_gene = deepcopy(parent1_gene)
                
                for gene in parent2.nodes_genes:
                    if(gene.id == parent1_gene.id):
                        if(np.random.random()<0.5):
                            children_gene.activation_function = gene.activation_function
                            break 
                children._add_copy_node(children_gene)

        for parent2_gene in parent2.nodes_genes:
            if((parent2_gene.id in nodes_ids) and (parent2_gene.id not in used)):

                used.append(parent2_gene.id)
                children_gene = deepcopy(parent2_gene)
                children._add_copy_node(children_gene)

        for node in parent1.nodes_genes:
            if((node.type != HIDDEN_TYPE) and (node.id not in used)):
                used.append(node.id)
                children_gene = deepcopy(node)
                children._add_copy_node(children_gene)

        return children
    

    @staticmethod
    def distance_genomes(genome1, genome2):
        len1 = len(genome1.connection_genes)
        len2 = len(genome1.connection_genes)


        if(len1 > len2):
            d = 0
            disjoint, excess, weight_diff = genome1.compare_all(genome2)
            d += genome1.config.C1 * disjoint
            d += genome1.config.C2 * excess
            d += genome1.config.C3 * weight_diff

            return d

        else:
            d = 0

            disjoint, excess, weight_diff = genome2.compare_all(genome1)
            d += genome1.config.C1 * disjoint
            d += genome1.config.C2 * excess
            d += genome1.config.C3 * weight_diff

            return d
   
    # @staticmethod
    # def save(genome,filename):

    #     pass



    # @staticmethod
    # def load(filename):


    #     return genome