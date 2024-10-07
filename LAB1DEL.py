#!/usr/bin/env python
# coding: utf-8

# In[1]:


#create and implement a basic neuron model with a computational framework. Integrate essential elements like input node, weight, parameters bias and an activation function(included but not limited to sigmoid, hyperbolic tangent, Relu) to construct a comprehensive representation of neuron
#evaluate and observe how each activation function influences network's behavior and effectiveness 
#handling different types of data


# In[4]:


import numpy as np

class Neuron:
    def __init__(self, num_inputs, activation_function='sigmoid'):
        self.weights = np.random.rand(num_inputs)
        self.bias = np.random.rand(1)
        self.activation_function = activation_function

    def activate(self, x):
        if self.activation_function == 'sigmoid':
            return 1 / (1 + np.exp(-x))
        elif self.activation_function == 'tanh':
            return np.tanh(x)
        elif self.activation_function == 'relu':
            return np.maximum(0, x)
        else:
            raise ValueError("Unsupported activation function")

    def forward(self, inputs):
        z = np.dot(inputs, self.weights) + self.bias
        return self.activate(z)


# In[2]:


def evaluate_neuron(activation_function):
    neuron = Neuron(num_inputs=3, activation_function=activation_function)
    sample_inputs = np.array([0.5, -1.5, 2.0])
    output = neuron.forward(sample_inputs)
    print(f"Activation Function: {activation_function}, Output: {output}")
activation_functions = ['sigmoid', 'tanh', 'relu']

for func in activation_functions:
    evaluate_neuron(func)


# In[6]:


import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0], [1], [1], [0]])

def build_and_train_model(hidden_activation, output_activation):
    model = Sequential()
    model.add(Dense(2, input_dim=2, activation=hidden_activation))
    model.add(Dense(1, activation=output_activation))
    model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.1), metrics=['accuracy'])
    model.fit(X, y, epochs=5000, verbose=0)
    return model

model_relu_sigmoid = build_and_train_model('relu', 'sigmoid')
print("ReLU Activation (Hidden Layer) and Sigmoid Activation (Output Layer) Results:")
predictions_relu_sigmoid = model_relu_sigmoid.predict(X)
print(np.round(predictions_relu_sigmoid))
model_tanh_sigmoid = build_and_train_model('tanh', 'sigmoid')
print("\nTanh Activation (Hidden Layer) and Sigmoid Activation (Output Layer) Results:")
predictions_tanh_sigmoid = model_tanh_sigmoid.predict(X)
print(np.round(predictions_tanh_sigmoid))


# In[5]:


import numpy as np
import matplotlib.pyplot as plt

def get_activation_function(name):
    if name == 'sigmoid':
        return lambda x: 1 / (1 + np.exp(-x))
    elif name == 'tanh':
        return lambda x: np.tanh(x)
    elif name == 'relu':
        return lambda x: np.maximum(0, x)

def plot_activation_function(activation_func, x_range, title):
    x = np.linspace(x_range[0], x_range[1], 400)
    y = activation_func(x)
    
    plt.plot(x, y, label=title)
    plt.title(title)
    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.grid()
    plt.legend()

x_range = (-5, 5)

plt.figure(figsize=(8, 4))

sigmoid = get_activation_function('sigmoid')
plot_activation_function(sigmoid, x_range, 'Sigmoid Activation Function')

tanh = get_activation_function('tanh')
plot_activation_function(tanh, x_range, 'Hyperbolic Tangent (Tanh) Activation Function')

relu = get_activation_function('relu')
plot_activation_function(relu, x_range, 'ReLU Activation Function')

plt.tight_layout()
plt.show()


# In[ ]:




