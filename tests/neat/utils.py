
import numpy as np

BIAS_TYPE = "bias"
INPUT_TYPE = "input"
OUTPUT_TYPE = "output"
HIDDEN_TYPE = "hidden"


def relu(x):
    return max(0,x)

RELU = relu




def sigmoid(x):
    return 1/(1+np.exp(-x))

SIGMOID = sigmoid




def tanh(x):
    return (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))

TANH = tanh


NAMES = {TANH: "tanh", SIGMOID: "sigmoid" , RELU: "relu"}

COLORS = {TANH: "red", SIGMOID: "green" , RELU: "yellow"}