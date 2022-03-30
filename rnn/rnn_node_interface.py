import abc, math, numpy as np
from rnn_genome import write_binary_string


NUMBER_NODE_TYPES: int = 9
NODE_TYPES = ["simple", "jordan", "elman", "UGRNN", "MGU", "GRU", "delta", "LSTM", "ENARC", "ENAS_DAG"]

def bound(value: float) -> float:
    if value < -10.0:
        value = -10.0
    elif value > 10.0:
        value = 10.0
    return value

def sigmoid(value: float) -> float:
    exp_value = math.exp(-value)
    return 1.0 / (1.0 + exp_value)

def sigmoif_derivatve(input: float) -> float:
    return input * (1 - input)

def identity(value: float) -> float:
    return value

def identity_derivative() -> float:
    return 1.0

def tanh_derivative(input: float) -> float:
    return 1 - (input ** 2)

def swish(value: float) -> float:
    exp_value = math.exp(-value)
    return value * (1.0 / (1.0 + exp_value))

def swish_derivative(value: float, input: float) -> float:
    sigmoid_value = sigmoid(value)
    return sigmoid_value + (input * (1 - sigmoid_value))

def leakyReLU(value: float) -> float:
    alpha = 0.01
    return np.fmax(alpha * value, value)

def leakyReLU_derivative(input: float) -> float:
    alpha = 0.01
    if input > 0:
        return 1
    return alpha

class RNN_Node_Interface(abc.ABC):

    _innovation_number: int = -1
    _layer_type: int = -1
    _node_type: int = -1

    _depth: float = -1.0

    _parameter_name: str = ""

    _enabled: bool = False
    _backward_reachable: bool = False
    _forward_reachable: bool = False

    _series_length: int = -1

    _input_values: [float] = []
    _output_values: [float] = []
    _error_values: [float] = []
    _d_input: [float] = []

    _inputs_fired: [int] = []
    _outputs_fired: [int] = []

    _total_inputs: int = -1
    _total_outputs: int = -1


    def __init__(self, _innovation_number: int, _layer_type: int, _depth: float, _parameter_name: str = None):
        self._innovation_number = _innovation_number
        self._layer_type = _layer_type
        self._depth = _depth
        self._total_inputs = 0
        self._enabled = True

        if self._layer_type != 1: # HIDDEN_LAYER = 1
            exit(1)

        if _parameter_name is not None: # If it's an input or output node
            self._parameter_name = _parameter_name
            if self._layer_type == 2: # OUTPUT_LAYER = 2
                self._total_outputs = 1
            else:
                self._total_outputs = 0
                self._total_inputs = 1

    def __del__(self):
        pass

    def get_node_type(self) -> int:
        return self._node_type

    def get_layer_type(self) -> int:
        return self._layer_type

    def get_innovation_number(self) -> int:
        return self._innovation_number

    def get_total_input(self) -> int:
        return self._total_inputs

    def get_total_outputs(self) -> int:
        return self._total_outputs

    def get_depth(self) -> float:
        return self._depth

    def is_reachable(self) -> bool:
        return self._forward_reachable and self._backward_reachable

    def is_enabled(self) -> bool:
        return self._enabled

    @abc.abstractmethod
    def initialize_lamarckian(self, mu, sigma):
        pass

    @abc.abstractmethod
    def initialize_xavier(self, rng_1_1, range):
        pass

    @abc.abstractmethod
    def initialize_kaiming(self, range):
        pass

    @abc.abstractmethod
    def initialize_uniform_random(self, rng):
        pass

    @abc.abstractmethod
    def input_fired(self, time, incoming_output):
        pass

    @abc.abstractmethod
    def output_fired(self, time, delta):
        pass

    @abc.abstractmethod
    def error_fired(self, time, error):
        pass

    @abc.abstractmethod
    def get_number_weights(self):
        pass

    @abc.abstractmethod
    def get_weights(self, parameters, offset=None):
        pass

    @abc.abstractmethod
    def set_weights(self, parameters, offset=None):
        pass

    @abc.abstractmethod
    def reset(self, series_length):
        pass

    @abc.abstractmethod
    def get_gradients(self, gradients):
        pass

    @abc.abstractmethod
    def copy(self):
        pass

    def write_to_stream(self, out):
        print(self._innovation_number)
        print(self._layer_type)
        print(self._node_type)
        print(self._depth)
        print(self._enabled)

        write_binary_string(self._parameter_name, "parameter_name")
















