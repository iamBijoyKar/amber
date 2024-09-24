import unittest

import numpy.testing as npt
from amber.neurons import Neuron
from amber.layers import InputLayer, Dense, SoftMax
from amber.models import Model


class NeuronTests(unittest.TestCase):
    def setUp(self):
        """Neuron setup with weights and bias
        """
        self.neuron = Neuron(weights=[1,2,3,4],bias=1.3,activation='relu')
        self.X = [4,3,2,1]
        self.result = 21.3

    def tearDown(self):
        """Deletion of Neuron instance 
        """
        del self.neuron

    def test_neuron_bias(self):
        self.assertEqual(self.neuron.bias, 1.3, 'bias is incorrect')
    
    def test_neuron_weights(self):
        npt.assert_array_equal(self.neuron.weights, [1,2,3,4], 'incorrect weights')

    def test_neuron_activation(self):
        self.assertEqual(self.neuron.activation, 'relu', 'incorrect activation')

    def test_neuron_forward(self):
        self.assertEqual(self.neuron.forward(self.X),self.result,'incorrect activation value returned')

        



if __name__ == '__main__':
    unittest.main()