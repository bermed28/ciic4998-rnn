from rnn_node_interface import RNN_Node_Interface
from rnn_edge import RNN_Edge
from rnn_recurrent_edge import RNN_Recurrent_Edge
from time_series import TimeSeriesSets
from word_series import Corpus

from datetime import datetime as dt
import math
import numpy as np

class RNN:

    #Private fields
    __series_length: int = -1
    __use_regression: bool = False

    __input_nodes: [RNN_Node_Interface] = []
    __output_nodes: [RNN_Node_Interface] = []
    __nodes: [RNN_Node_Interface] = []
    __edges: [RNN_Edge] = []
    __recurrent_edges: [RNN_Recurrent_Edge] = []

    def __init__(self, nodes: [RNN_Node_Interface], edges: [RNN_Edge], input_parameter_names: [str], output_parameter_names: [str], recurrent_edges: [RNN_Recurrent_Edge] = None):
        if recurrent_edges is None:
            self.__nodes = nodes
            self.__edges = edges

            self.__edges.sort(key= lambda e: e.get_input_node().get_depth())

            for node in self.__nodes:
                if node.layer_type == 0: #input layer
                    self.__input_nodes.append(node)
                elif node.layer_type == 2: #output layer
                    self.__output_nodes.append(node)

            self.fix_parameter_orders(input_parameter_names, output_parameter_names)
            self.validate_parameters(input_parameter_names, output_parameter_names)
        else:
            self.__nodes = nodes
            self.__edges = edges
            self.__recurrent_edges = recurrent_edges

            print(f"Creating rnn with {len(nodes)} nodes, {len(edges)} edges") #change to Log.debug later

            for node in self.__nodes:
                if node.layer_type == 0:  # input layer
                    self.__input_nodes.append(node)
                    print("had input node!")
                elif node.layer_type == 2:  # output layer
                    self.__output_nodes.append(node)
                    print("had output node!")

            print(f"fixing parameter orders, len(__input_nodes): {len(self.__input_nodes)}")  #change to Log.debug later
            self.fix_parameter_orders(input_parameter_names, output_parameter_names)
            print(f"validating parameters, len(__input_nodes): {len(self.__input_nodes)}")  #change to Log.debug later
            self.validate_parameters(input_parameter_names, output_parameter_names)

            print(f"got RNN with {len(self.__nodes)} nodes, {len(self.__edges)} edges, {len(self.__recurrent_edges)} recurrent edges")  #change to Log.trace later

    def __del__(self):
        node = None
        while len(self.__nodes) > 0:
            nodes = self.__nodes[-1]
            self.__nodes.pop()
            del node

        edge = None
        while len(self.__edges) > 0:
            edge = self.__edges[-1]
            self.__edges.pop()
            del edge

        recurrent_edge = None
        while len(self.__recurrent_edges) > 0:
            recurrent_edge = self.__recurrent_edges[-1]
            self.__recurrent_edges.pop()
            del recurrent_edge

        while len(self.__input_nodes) > 0:
            self.__input_nodes.pop()

        while len(self.__output_nodes) > 0:
            self.__output_nodes.pop()

    def fix_parameter_orders(self, input_parameter_names: [str], output_parameter_names: [str]) -> None:
        ordered_input_nodes = []

        for i in range(len(input_parameter_names)):
            for j in range(len(self.__input_nodes)):
                if self.__input_nodes[j].parameter_name == input_parameter_names[i]:
                    ordered_input_nodes.append(self.__input_nodes[j])
                    del self.__input_nodes[j]

        self.__input_nodes = ordered_input_nodes

        ordered_output_nodes = []

        for i in range(len(output_parameter_names)):
            for j in range(len(self.__output_nodes) - 1, -1, -1):
                if self.__output_nodes[j].parameter_name == output_parameter_names[i]:
                    ordered_output_nodes.append(self.__output_nodes[j])
                    del self.__output_nodes[j]

        self.__output_nodes = ordered_output_nodes

    def validate_parameters(self, input_parameter_names: [str], output_parameter_names: [str]) -> None:
        if len(self.__input_nodes) != len(input_parameter_names):
            exit(1)

        parameter_mismatch = False

        for i in range(len(self.__input_nodes)):
            if (self.__input_nodes[i].parameter_name != input_parameter_names[i]):
                parameter_mismatch = True

        if parameter_mismatch:
            exit(1)

        if len(self.__output_nodes) != len(output_parameter_names):
            exit(1)

        parameter_mismatch = False
        for i in range(len(self.__output_nodes)):
            if self.__output_nodes[i].parameter_name != output_parameter_names[i]:
                parameter_mismatch = True

        if parameter_mismatch:
            exit(1)

    def get_number_nodes(self) -> int:
        return len(self.__nodes)

    def get_number_edges(self) -> int:
        return len(self.__edges)

    def get_node(self, i) -> RNN_Node_Interface:
        if 0 <= i <= len(self.__nodes) - 1:
            return self.__nodes[i]

        return None

    def get_edge(self, i) -> RNN_Edge:
        if 0 <= i <= len(self.__edges) - 1:
            return self.__edges[i]

        return None

    def forwards_pass(self, series_data: [[float]], using_dropout: bool, training: bool, dropout_probability: float) -> None:
        self.__series_length = len(series_data[0])

        if len(self.__input_nodes) != len(series_data):
            print(F"ERROR: number of input nodes {len(self.__input_nodes)} != number of time series data input fields {len(series_data)}")
            for i in range(len(self.__nodes)):
                print(f"node[{i}], in: {self.__nodes[i].get_innovation_number()}, depth: {self.__nodes[i].get_depth()}, layer_type: {self.__nodes[i].get_layer_type()}, node_type: {self.__nodes[i].get_node_type()}")
            exit(1)

        for i in range(len(self.__nodes)):
            self.__nodes[i].reset(self.__series_length)

        for i in range(len(self.__edges)):
            self.__edges[i].reset(self.__series_length)

        for i in range(len(self.__recurrent_edges)):
            self.__recurrent_edges[i].reset(self.__series_length)

        for i in range(len(self.__recurrent_edges)):
            if self.__recurrent_edges[i].is_reachable():
                self.__recurrent_edges[i].first_propagate_backward()

        for time in range(self.__series_length - 1, -1, -1):

            for i in range(len(self.__input_nodes)):
                self.__input_nodes[i].input_fired(time, series_data[i][time])

            if using_dropout:
                for i in range(len(self.__edges) - 1, -1, -1):
                    if self.__edges[i].is_reachable():
                        self.__edges[i].propagate_forward(time, training, dropout_probability)
            else:
                for i in range(len(self.__edges) - 1, -1, -1):
                    if self.__edges[i].is_reachable():
                        self.__edges[i].propagate_forward(time)

            for i in range(len(self.__recurrent_edges) - 1, -1, -1):
                if self.__recurrent_edges[i].is_reachable():
                    self.__recurrent_edges[i].propagate_forwards(time)


    def backwards_pass(self, error: float, using_dropout: bool, training: bool, dropout_probability: float) -> None:
        for i in range(len(self.__recurrent_edges)):
            if self.__recurrent_edges[i].is_reachable():
                self.__recurrent_edges[i].first_propagate_backward()

        for time in range(self.__series_length - 1, -1 , -1):
            for i in range(len(self.__output_nodes)):
                self.__output_nodes[i].error_fired(time, error)
            if using_dropout:
                for i in range(len(self.__edges) - 1, -1, -1):
                    if self.__edges[i].is_reachable():
                        self.__edges[i].propagate_backward(time, training, dropout_probability)
            else:
                for i in range(len(self.__edges) - 1, -1, -1):
                    if self.__edges[i].is_reachable():
                        self.__edges[i].propagate_backward(time)

            for i in range(len(self.__recurrent_edges) - 1, -1, -1):
                if self.__recurrent_edges[i].is_reachable():
                    self.__recurrent_edges[i].propagate_backwards(time)

    def calculate_error_softmax(self, expected_outputs: [[float]]) -> float:
        cross_entropy_sum = 0.0
        softmax = 0.0

        for j in range(len(expected_outputs[0])):
            softmax_sum = 0.0
            cross_entropy = 0.0

            for i in range(len(self.__output_nodes)):
                softmax_sum += math.exp(self.__output_nodes[i].output_values[j])

            for i in range(len(self.__output_nodes)):
                softmax = math.exp(self.__output_nodes[i].ouput_values[j]) / softmax_sum
                error = softmax - expected_outputs[i][j]
                self.__output_nodes[i].error_values[j] = error

                cross_entropy = -expected_outputs[i][j] * math.log(softmax, base=math.e)

                cross_entropy_sum += cross_entropy

        return cross_entropy_sum

    def calculate_error_mse(self, expected_outputs: [[float]]) -> float:
        mse_sum = 0.0

        for i in range(len(self.__output_nodes)):
            mse = 0.0

            for j in range(len(expected_outputs[i])):
                error = self.__output_nodes[i].output_values[j] - expected_outputs[i][j]

                self.__output_nodes[i].error_value[j] = error
                mse += error ** 2

            mse_sum += mse / len(expected_outputs[i])

        return mse_sum

    def calculate_error_mae(self, expected_outputs: [[float]]) -> float:
        mae_sum = 0.0

        for i in range(len(self.__output_nodes)):
            mae = 0.0
            for j in range(len(expected_outputs[i])):
                error = np.fabs(self.__output_nodes[i].output_values[j] - expected_outputs[i][j])
                mae += error

                if error == 0:
                    error = 0
                else:
                    error = (self.__output_nodes[i].output_values[j] - expected_outputs[i][j]) / error

                self.__output_nodes[i].error_values[j] = error

            mae_sum += mae / len(expected_outputs[i])

        return mae_sum

    def prediction_softmax(self, series_data: [[float]], expected_output: [[float]], using_dropout: bool, training: bool, dropout_probability: float) -> float:
        self.forwards_pass(series_data, using_dropout, training, dropout_probability)
        return self.calculate_error_softmax(expected_output)

    def prediction_mse(self, series_data: [[float]], expected_output: [[float]], using_dropout: bool, training: bool, dropout_probability: float) -> float:
        self.forwards_pass(series_data, using_dropout, training, dropout_probability)
        return self.calculate_error_mse(expected_output)

    def prediction_mae(self, series_data: [[float]], expected_output: [[float]], using_dropout: bool, training: bool, dropout_probability: float) -> float:
        self.forwards_pass(series_data, using_dropout, training, dropout_probability)
        return self.calculate_error_mae(expected_output)

    def get_predictions(self, series_data: [[float]], expected_output: [[float]], using_dropout: bool, dropout_probability: float) -> [float]:
        self.forwards_pass(series_data, using_dropout, False, dropout_probability)
        result = []

        for j in range(self.__series_length):
            for i in range(len(self.__output_nodes)):
                result.append(self.__output_nodes[i].output_values[j])

        return result

    def write_predictions(self, output_filename: str, input_parameter_names: [str], output_parameter_names: [str], series_data: [[float]], expected_outputs: [[float]], using_dropout: bool, dropout_probability: float, time_series_sets: TimeSeriesSets = None, word_series_sets: Corpus = None) -> None:
        self.forwards_pass(series_data, using_dropout, False, dropout_probability)

        outfile = open(output_filename, 'w')
        outfile.write("#")

        for i in range (len(self.__input_nodes)):
            if i > 0:
                outfile.write(",")
            outfile.write(input_parameter_names[i])

        for i in range(len(self.__output_nodes)):
            outfile.write(",")
            outfile.write(f"expected_{output_parameter_names[i]}")

        for i in range(len(self.__output_nodes)):
            outfile.write(",")
            outfile.write(f"predicted_{output_parameter_names[i]}")

        outfile.write('\n')

        # for j in range(self.__series_length):
        #     for i in range(len(self.__input_nodes)):
        #         if i > 0:
        #             outfile.write(",")
        #         outfile.write(time_series_sets.denormalize(input_parameter_names[i], series_data[i][j]))
        #     for i in range(len(self.__output_nodes)):
        #         outfile.write(",")
        #         outfile.write(time_series_sets.denormalize(output_parameter_names[i], expected_outputs[i][j]))
        #     for i in range(len(self.__output_nodes)):
        #         outfile.write(",")
        #         outfile.write(time_series_sets.denormalize(output_parameter_names[i], self.__output_nodes[i].output_values[j]))
        #     outfile.write('\n')

        outfile.close()

    def initialize_randomly(self) -> None:
        number_of_weights = self.get_number_weights()
        parameters = [0 * number_of_weights]

        seed = dt.now().timestamp()
        for i in range(len(parameters)):
            parameters[i] = np.random.uniform(-0.5, 0.5)

        self.set_weights(parameters)


    def get_weights(self, parameters: [float]) -> None:
        parameters = self.resize(parameters, self.get_number_weights())

        current = 0

        for i in range(len(self.__nodes)):
            self.__nodes[i].get_weights(current, parameters)

        for i in range(len(self.__edges)):
            parameters[current] = self.__edges[i].weight
            current += 1

        for i in range(len(self.__recurrent_edges)):
            parameters[current] = self.__recurrent_edges[i].weight
            current += 1

    def set_weights(self, parameters: [float]) -> None:
        if(len(parameters) != self.get_number_weights()):
            print(f"ERROR! Trying to set weights where the RNN has {self.get_number_weights()} weights, and the parameters array has {len(parameters)} weights!")

    def get_number_weights(self):
        number_weights = 0

        for i in range(len(self.__nodes)):
            number_weights += self.__nodes[i].get_number_weights()

        for _ in self.__edges:
            number_weights += 1

        for _ in self.__recurrent_edges:
            number_weights += 1

        return number_weights

    def get_analytical_gradient(self, test_parameters: [float], inputs: [[float]], outputs: [[float]], mse: float, analytic_gradient: [float], using_dropout: bool, training: bool, dropout_probability: float) -> None:
        analytic_gradient = [0.0 for _ in range(len(test_parameters))]

        self.set_weights(test_parameters)
        self.forwards_pass(inputs, using_dropout, training, dropout_probability)

        if self.__use_regression:
            mse = self.calculate_error_mse(outputs)
            self.backwards_pass(mse * (1.0 / len(outputs[0])) * 2.0, using_dropout, training, dropout_probability)
        else:
            self.backwards_pass(mse * (1.0 / len(outputs[0])), using_dropout, training, dropout_probability)

        current_gradients = []

        current = 0
        for i in range(len(self.__nodes)):
            if self.__nodes[i].is_reachable():
                self.__nodes[i].get_gradients(current_gradients)

                for j in range(len(current_gradients)):
                    analytic_gradient[current] = current_gradients[j]
                    current += 1

        for i in range(len(self.__edges)):
            if self.__edges[i].is_reachable():
                analytic_gradient[current] = self.__edges[i].get_gradient()
                current += 1

        for i in range(len(self.__recurrent_edges)):
            if self.__recurrent_edges[i].is_reachable():
                analytic_gradient[current] = self.__recurrent_edges[i].get_gradient()
                current += 1

    def get_empirical_gradient(self, test_parameters: [float], inputs: [[float]], outputs: [[float]], mse: float, empirical_gradient: [float], using_dropout: bool, training: bool, dropout_probability: float) -> None:
        empirical_gradient = [0.0 for _ in range(len(test_parameters))]
        deltas = []

        self.set_weights(test_parameters)
        self.forwards_pass(inputs,using_dropout,training, dropout_probability)
        original_mse = self.calculate_error_mse(outputs)

        diff = 0.00001
        mse1, mse2, save = 0.0, 0.0, 0.0

        parameters = test_parameters
        for i in range(len(parameters)):
            save = parameters[i]
            parameters[i] = save - diff
            self.set_weights(parameters)
            self.forwards_pass(inputs, using_dropout, training, dropout_probability)
            self.get_mse(outputs, mse1, deltas)

            parameters[i] = save + diff
            self.set_weights(parameters)
            self.forwards_pass(inputs, using_dropout, training, dropout_probability)
            self.get_mse(outputs, mse2, deltas)

            empirical_gradient[i] = (mse2 - mse1) / (2.0 * diff)
            empirical_gradient[i] *= original_mse

            parameters[i] = save

        mse = original_mse


    def get_mse(self, expected: [[float]], mse: float, deltas: [[float]]):
        pass #port over to mse.py later

    def get_mae(self, expected: [[float]], mae: float, deltas, output_values: [float] = None, mae_sum: float = None):
        #port to mse.py later on
        if output_values is not None and mae_sum is None and isinstance(deltas, list):
            deltas = [0.0 for _ in range(len(expected))]

            mae = 0.0
            error = 0.0

            for j in range(len(expected)):
                error = math.fabs(output_values[j] - expected[j])
                if error == 0:
                    deltas[j] = 0
                else:
                    deltas[j] = (output_values[j] - expected[j]) / error

                mae += error
            mae /= len(expected)

            d_mae = mae * (1.0 / len(expected))
            for j in range(len(expected)):
                deltas[j] *= d_mae
        else:
            deltas = [[0.0 for _ in range(len(expected[0]))] for _ in range(len(expected))]

            mae_sum = 0.0
            mae = 0.0
            error = 0.0

            for i in range(len(self.__output_nodes)):
                mae = 0.0
                for j in range(len(self.__output_nodes)):
                    error = math.fabs(self.__output_nodes[i].output_values[j] - expected[i][j])
                    if error == 0:
                        deltas[i][j] = 0
                    else:
                        deltas[i][j] = (self.__output_nodes[i].output_values[j] - expected[i][j]) / error
                    mae += error
                mae /= len(expected[i])
                mae_sum += mae

            d_mae = mae * (1.0 / len(expected[0]))
            for i in range(len(self.__output_nodes)):
                for j in range(len(expected[i])):
                    deltas[i][j] *= d_mae


    def enable_use_regression(self, _use_regression: bool):
        self.__use_regression = _use_regression

    def resize(self, array, newSize):
        newArr = [0 for _ in range(newSize)]

        for i in range(len(array)):
            newArr[i] = array[i]

        return newArr



