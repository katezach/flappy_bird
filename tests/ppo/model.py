import tensorflow as tf
from tensorflow import keras
class N_Network():
    def __init__(self, hidden_layers = [256], hidden_activation= tf.nn.relu, type= "policy"):
        self.hidden_layers = hidden_layers
        self.hidden_activation = hidden_activation
        if(type == "policy"):
            self.final_activation = tf.nn.softmax
        elif(type == "value"):
            self.final_activation = None


    def create_model(self, input_shape, no_outputs):
        input = tf.keras.Input(input_shape)
        layer = tf.keras.layers.Dense(self.hidden_layers[0],kernel_initializer= tf.keras.initializers.HeUniform(), activation=self.hidden_activation)(input)
        for no_neurons in self.hidden_layers[1:]:
            layer = tf.keras.layers.Dense(no_neurons,kernel_initializer= tf.keras.initializers.HeUniform(), activation=self.hidden_activation)(layer)   
        output = tf.keras.layers.Dense(no_outputs, activation=self.final_activation)(layer)
        return tf.keras.Model(inputs=input, outputs=output)

class Conv_Network():
    def __init__(self, hidden_layers = [32], hidden_activation= tf.nn.relu, type= "policy"):
        self.hidden_layers = hidden_layers
        self.hidden_activation = hidden_activation
        if(type == "policy"):
            self.final_activation = tf.nn.softmax
        elif(type == "value"):
            self.final_activation = None
    

    def create_model(self,no_inputs, no_outputs):
        
        input = tf.keras.Input(no_inputs)
        layer = tf.keras.layers.Conv2D(self.hidden_layers[0], (3,3), padding="same",activation=self.hidden_activation,kernel_initializer= tf.keras.initializers.HeUniform(),)(input)
        layer = tf.keras.layers.MaxPool2D((3,3))(layer)
        for no_features in self.hidden_layers[1:]:
            layer = tf.keras.layers.Conv2D(no_features, (3,3), padding="same", activation=self.hidden_activation,kernel_initializer= tf.keras.initializers.HeUniform(),)(layer)   
            layer = tf.keras.layers.MaxPool2D((3,3))(layer)
        flatten = tf.keras.layers.Flatten()(layer)
        output = tf.keras.layers.Dense(no_outputs, activation=self.final_activation)(flatten)
        
        return tf.keras.Model(inputs=input, outputs=output)

      
