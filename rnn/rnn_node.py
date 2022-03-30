from rnn_node_interface import RNN_Node_Interface
from rnn_node_interface import bound, tanh_derivative
import numpy as np
import math


class RNN_Node(RNN_Node_Interface):

    def __init__(self, innovation_number, layer_type, depth, node_type, parameter_name=None):
        RNN_Node_Interface.__init__(innovation_number, layer_type, depth, parameter_name)
        self.__bias = 0
        self.__d_bias = -1
        self._node_type = node_type
        self.__ld_output = []


    def initialize_lamarckian(self, mu, sigma):
        self.__bias = bound(np.random.uniform(mu, sigma))


    def initialize_xavier(self, rng_1_1, range):
        self.__bias = range * (rng_1_1())

    def initialize_kaiming(self, range):
        self.__bias = range * np.random.uniform(0, 1)

    def initialize_uniform_random(self, rng):
        self.__bias = np.random.uniform()

    def input_fired(self, time, incoming_output):
        self._inputs_fired[time] += 1
        self._input_values[time] += incoming_output
        if self._inputs_fired[time] < self._total_inputs:
            return
        elif self._inputs_fired[time] > self._total_inputs:
            exit(1)
        self._output_values[time] = math.tanh(self._input_values[time] + self.__bias)
        self.__ld_output[time] = tanh_derivative(self._output_values[time])

    def try_update_deltas(self, time):
        if self._outputs_fired[time] < self._total_outputs:
            return
        elif self._outputs_fired[time]> self._total_outputs:
            exit(1)

        self._d_input[time] *= self.__ld_output[time]
        self.__d_bias += self._d_input[time]

    def output_fired(self, time, delta):
        self._outputs_fired[time] += 1
        self._d_input[time] += delta
        self.try_update_deltas(time)

    def error_fired(self, time, error):
        self._outputs_fired[time] += 1
        self._d_input[time] += self._error_values[time] * error
        self.try_update_deltas(time)

    def get_number_weights(self):
        return 1

    def get_weights(self, parameters, offset=None):
        if offset is None:
            offset = 0
            self.get_weights(offset, parameters)
        else:
            parameters[offset] = self.__bias
            offset += 1

    def set_weights(self, parameters, offset=None):
        self.__bias = bound(parameters[offset])
        offset += 1

    def reset(self, series_length):
        self._series_length = series_length
        self.__ld_output = [0.0] * series_length
        self._d_input = [0.0] * series_length
        self._input_values = [0.0] * series_length
        self._output_values = [0.0] * series_length
        self._error_values = [0.0] * series_length
        self._inputs_fired = [0.0] * series_length
        self._outputs_fired = [0.0] * series_length
        self.__d_bias = 0.0


    def get_gradients(self, gradients):
        for i in range(0, 1):
            gradients[i] = self.__d_bias

    def copy(self):
        n = None
        if self._layer_type == 1: # HIDDEN_LAYER
            n = RNN_Node(self._innovation_number, self._layer_type, self._depth, self._node_type)
        else :
            n = RNN_Node(self._innovation_number, self._layer_type, self._depth, self._node_type, self._parameter_name)

        #Copy RNN_Node Values
        n.__d_bias = self.__d_bias
        n.__bias = self.__bias
        n.__ld_output = self.__ld_output

        #Copy RNN_Node_Interface Values
        n._series_length = self._series_length
        n._input_values = self._input_values
        n._output_values = self._output_values
        n._error_values = self._error_values
        n._d_input = self._d_input
        n._inputs_fired = self._inputs_fired
        n._total_inputs = self._total_inputs
        n._outputs_fired = self._outputs_fired
        n._total_outputs = self._total_outputs
        n._enabled = self._enabled
        n._forward_reachable = self._forward_reachable
        n._backward_reachable = self._backward_reachable
