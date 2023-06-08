from utils import RELU,NAMES


class Node:
    def __init__(self,node_id,node_type, activation_function = RELU) -> None:
        self.id = node_id
        self.type = node_type
        self.activation_function = activation_function
    
    def __str__(self) -> str:
        return str(self.id) + ": " + self.type + " activ f: " +  NAMES[self.activation_function]
    
    
