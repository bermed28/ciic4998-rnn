from rnn_node_interface import RNN_Node_Interface
class RNN_Recurrent_Edge():

    __series_length = 0
    __outputs: [float] = []
    __deltas: [float] = []
    __dropped_out: [float] = []

    __weight: float = 0.0
    __d_weight: float = 0.0

    def __init__(self, innovation_number, recurrent_depth,  input_node: RNN_Node_Interface= None, output_node:RNN_Node_Interface = None, input_innovation_number = -1, output_innovation_number = -1, nodes: [RNN_Node_Interface] = None):
        self.__innovation_number = innovation_number
        self.__recurrent_depth = recurrent_depth
        if input_node is not None and output_node is not None:

            if recurrent_depth <= 0:
                exit(1)

            self.__input_node: RNN_Node_Interface = input_node
            self.__output_node: RNN_Node_Interface = output_node

            self.__enabled = True
            self.__forward_reachable = False
            self.__backward_reachable = False

            self.__input_innovation_number = self.__input_node.get_innovation_number()
            self.__output_innovation_number = self.__output_node.get_innovation_number()

            self.__input_node._total_outputs += 1
            self.__output_node._total_inputs += 1

        else:
            self.__input_innovation_number = input_innovation_number
            self.__output_innovation_number = output_innovation_number

            for i in range(len(nodes)):
                if nodes[i]._innovation_number == input_innovation_number:
                    if self.__input_node is None:
                        exit(1)
                    self.__input_node = nodes[i]

                if nodes[i]._innovation_number == output_innovation_number:
                    if self.__output_node is None:
                        exit(1)
                    self.__output_node = nodes[i]

            if self.__input_node is None or self.__output_node is None:
                exit(1)

    def copy(self, nodes):
        e = RNN_Recurrent_Edge(innovation_number=self.__innovation_number, input_innovation_number=self.__input_innovation_number, output_innovation_number=self.__output_innovation_number, nodes=nodes)
        e.__recurrent_depth = self.__recurrent_depth

        e.__weight = self.__weight
        e.__d_weight = self.__d_weight

        e.__outputs = self.__outputs
        e.__deltas = self.__deltas

        e.__enabled = self.__enabled
        e.__forward_reachable = self.__forward_reachable
        e.__backward_reachable = self.__backward_reachable

        return e

    def reset(self, series_length):
        self.__series_length = series_length
        self.__d_weight = 0.0
        self.__outputs = [0.0] * series_length
        self.__deltas = [0.0] * series_length
        self.__dropped_out = [0.0] * series_length

    def first_propagate_forward(self):
        for i in range(self.__recurrent_depth):
            self.__output_node.input_fired(i, 0.0)

    def propagate_forward(self, time):
        if self.__input_node._inputs_fired[time] != self.__input_node._total_inputs:
            exit(1)

        output = self.__input_node._output_values[time] * self.__weight

        if time < self.__series_length - self.__recurrent_depth:
            self.__outputs[time + self.__recurrent_depth] = output
            self.__output_node.input_fired(time=time + self.__recurrent_depth, incoming_output=output)

    def first_propagate_backward(self):
        for i in range(self.__recurrent_depth):
            self.__input_node.output_fired(self.__series_length - 1 - i, 0.0)

    def propagate_backward(self, time: int):
        if self.__output_node._outputs_fired[time] != self.__output_node._total_outputs:
            exit(1)
        delta = self.__output_node._d_input[time]

        if time - self.__recurrent_depth >= 0:
            self.__d_weight += delta * self.__input_node._output_values[time - self.__recurrent_depth]
            self.__deltas[time] = delta * self.__weight
            self.__input_node.output_fired(time=time - self.__recurrent_depth, delta=self.__deltas[time])

    def get_recurrent_depth(self):
        return self.__recurrent_depth

    def get_gradient(self):
        return self.__d_weight

    def get_innovation_number(self):
        return self.__innovation_number

    def get_input_innovation_number(self):
        return self.__input_innovation_number

    def get_output_innovation_number(self):
        return self.__output_innovation_number

    def get_input_node(self):
        return self.__input_node

    def get_output_node(self):
        return self.__output_node

    def is_enabled(self):
        return self.__enabled

    def is_reachable(self):
        return self.__forward_reachable and self.__backward_reachable

    def equals(self, other):
        return self.__innovation_number == other.__innovation_number and self.__enabled == other.__enabled

    def write_to_stream(self):
        pass