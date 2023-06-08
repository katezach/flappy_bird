
from utils import RELU

class NeuralNet:
    def __init__(self,inputs,outputs,layers) -> None:
        self.inputs = inputs 
        self.outputs = outputs
        self.layers = layers
        

    def predict(self, inputs):
        for input, node in zip(inputs,self.inputs):
            node.set_value(input)

        

        for layer in self.layers:
            for node in layer:
                node.forward()

        out = [out.value for out in self.outputs]
        return out
        
    

        



class Unit:
    def  __init__(self, id, previous, weights, activation_function = RELU) -> None:
        self.id = id
        self.previous = previous
        self.weights = weights
        self.value = 1
        self.activation_function = activation_function

    def set_value(self, value):
        self.value = value

    def forward(self):
        self.value = 0
        for prev,weight in zip(self.previous,self.weights):
            self.value += prev.value * weight
        #apply some function
        self.value =  self.activation_function(self.value)

    def __str__(self) -> str:
        return str(self.id)
    

