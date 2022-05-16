import time
import math
from weight_initialize import WeightType
from random import randint as random
from loguru import logger as log
import numpy as np
from rnn_node_interface import bound
from rnn.rnn import RNN


EXAMM_MAX_DOUBLE = 10000000

class RNN_Genome:

    def __init__(self, nodes, edges, recurrent_edges, weight_initialize, weight_inheritance, mutated_component_weight, seed = None):
        self.structural_hash = ""

        self.generated_by_map = {}
        self.initial_parameters = []
        self.best_parameters = []
        self.normal_distribution = []  # np.random.normal
        self.normalize_type = ""
        self.normalize_mins = {}
        self.normalize_maxs = {}
        self.normalize_avgs = {}
        self.normalize_std_devs = {}

        self.tl_with_epigenetic = False

        self.input_parameter_names = []
        self.output_parameter_names = []

        self.generation_id = -1
        self.group_id = -1

        self.best_validation_mse = EXAMM_MAX_DOUBLE
        self.best_validation_mae = EXAMM_MAX_DOUBLE

        self.nodes = nodes
        self.edges = edges
        self.recurrent_edges = recurrent_edges
        self.weight_initialize = weight_initialize
        self.weight_inheritance = weight_inheritance
        self.mutated_component_weight = mutated_component_weight

        self.sort_nodes_by_depth()
        self.sort_edges_by_depth()

        self.bp_iterations = 20000
        self.learning_rate = 0.001
        self.adapt_learning_rate = False
        self.use_nesterov_momentum = True
        self.use_reset_weights = False

        self.use_high_norm = True
        self.high_threshold = 1.0
        self.use_low_norm = True
        self.low_threshold = 0.5

        self.use_dropout = False
        self.dropout_probability = 0.5

        self.use_regression = False

        self.log_filename = ""

        if seed is not None:
            self.seed = time.time()
        self.seed = seed

        self.rng = np.random.uniform(-0.5, 0.5)
        self.rng_0_1 = np.random.uniform(0.0, 1.0)
        self.rng_1_1 = np.random.uniform(-1.0, 1.0)

        self.assign_reachability()

    def __del__(self):
        while len(self.nodes) > 0:
            node = self.nodes.pop()
            del node

        while len(self.edges) > 0:
            edge = self.edges.pop()
            del edge

        while len(self.recurrent_edges) > 0:
            recurrent_edge = self.recurrent_edges.pop()
            del recurrent_edge

    def set_parameter_names(self, input_parameter_names, output_parameter_names):
        self.input_parameter_names = input_parameter_names
        self.output_parameter_names = output_parameter_names

    def copy(self):
        node_copies = [node.copy() for node in self.nodes]
        edge_copies = [edge.copy() for edge in self.edges]
        recurrent_edge_copies = [recurrent_edge.copy() for recurrent_edge in self.recurrent_edges]

        other = RNN_Genome(node_copies, edge_copies, recurrent_edge_copies, self.weight_initialize, self.weight_inheritance, self.mutated_component_weight)

        other.group_id = self.group_id
        other.bp_iterations = self.bp_iterations
        other.learning_rate = self.learning_rate
        other.use_nesterov_momentum = self.use_nesterov_momentum
        other.use_high_norm = self.use_high_norm
        other.high_threshold = self.high_threshold
        other.use_low_norm = self.use_low_norm
        other.low_threshold = self.low_threshold

        other.use_dropout = self.use_dropout
        other.dropout_probability = self.dropout_probability

        other.log_filename = self.log_filename

        other.generated_by_map = self.generated_by_map

        other.initial_parameters = self.initial_parameters

        other.best_validation_mse = self.best_validation_mse
        other.best_validation_mae = self.best_validation_mae
        other.best_parameters = self.best_parameters

        other.input_parameter_names = self.input_parameter_names
        other.output_parameter_names = self.output_parameter_names

        other.normalize_type = self.normalize_type
        other.normalize_mins = self.normalize_mins
        other.normalize_maxs = self.normalize_maxs
        other.normalize_avgs = self.normalize_avgs
        other.normalize_std_devs = self.normalize_std_devs

        return other


    def get_avg_recurrent_depth(self):
        count = 0
        avg = 0.0

        for edge in self.recurrent_edges:
            if edge.is_reachable():
                avg += edge.get_recurrent_depth()
                count += 1

        if count == 0:
            return 0

        return avg / count

    def get_edge_count_str(self, recurrent):
        if recurrent:
            return f"{self.get_enabled_recurrent_edge_count()} ({len(self.recurrent_edges)})"
        else:
            return f"{self.get_enabled_edge_count()} ({len(self.edges)})"

    def get_node_count_str(self, node_type):
        if node_type < 0:
            return f"{self.get_enabled_node_count()} ({self.get_node_count()})"
        else:
            enabled_nodes = self.get_enabled_node_count(node_type)
            total_nodes = self.get_node_count(node_type)

            if total_nodes > 0:
                return f"{enabled_nodes} ({total_nodes})"

        return ""

    def get_enabled_node_count(self, node_type=None):
        count = 0
        for node in self.nodes:
            if node_type is not None:
                if node.enabled and node.layer_type == 1 and node.node_type == node_type:
                    count += 1
            else:
                if node.enabled:
                    count += 1
        return count

    def get_node_count(self, node_type=None):
        if node_type is not None:
            count = 0
            for node in self.nodes:
                if node.node_type == node_type:
                    count += 1

            return count

        return len(self.nodes)

    def clear_generated_by(self):
        self.generated_by_map.clear()

    def update_generation_map(self, generation_map):
        for k, v in self.generated_by_map.items():
            generation_map[k] += v

    def generated_by_string(self):
        result = "["
        first = True
        for k, v in self.generated_by_map.items():
            if not first:
                result += ", "
            result += f"{k}:{v}"
            first = False

        result += "]"
        return result

    def get_generated_by_map(self):
        return self.generated_by_map

    def get_input_parameter_names(self):
        return self.input_parameter_names

    def get_output_parameter_names(self):
        return self.output_parameter_names

    def set_normalize_bounds(self, normalize_type, normalize_mins, normalize_maxs, normalize_avgs, normalize_std_devs):
        self.normalize_type = normalize_type
        self.normalize_mins = normalize_mins
        self.normalize_maxs = normalize_maxs
        self.normalize_avgs = normalize_avgs
        self.normalize_std_devs = normalize_std_devs

    def get_normalize_type(self):
        return self.normalize_type

    def get_normalize_mins(self):
        return self.normalize_mins

    def get_normalize_maxs(self):
        return self.normalize_maxs

    def get_normalize_avgs(self):
        return self.normalize_avgs

    def get_normalize_std_devs(self):
        return self.normalize_std_devs

    def get_group_id(self):
        return self.group_id

    def set_group_id(self, group_id):
        self.group_id = group_id

    def get_enabled_edge_count(self):
        count = 0
        for edge in self.edges:
            if edge.enabled:
                count += 1
        return count

    def get_enabled_recurrent_edge_count(self):
        count = 0
        for edge in self.recurrent_edges:
            if edge.enabled:
                count += 1
        return count

    def set_bp_ierations(self, bp_iterations, epochs_acc_freq):
        if epochs_acc_freq > 0:
            if self.generation_id < epochs_acc_freq:
                self.bp_iterations = 0
            else:
                n = math.floor(self.generation_id / epochs_acc_freq) - 1
                self.bp_iterations = math.pow(2, n)

            log.info(f"Setting up iteration {self.bp_iterations} to genome {self.generation_id}")
        else:
            self.bp_iterations = bp_iterations

    def get_bp_iterations(self):
        return self.bp_iterations

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def set_nesterov_momentum(self, use_nesterov_momentum):
        self.use_nesterov_momentum = use_nesterov_momentum

    def disable_high_threshold(self):
        self.use_high_norm = False

    def enable_high_threshold(self, high_threshold):
        self.use_high_norm = False
        self.high_threshold = high_threshold

    def disable_low_threshold(self):
        self.use_low_norm = False

    def enable_low_threshold(self, low_threshold):
        self.use_low_norm = False
        self.low_threshold = low_threshold

    def disable_dropout(self):
        self.use_dropout = False
        self.dropout_probability = 0

    def enable_dropout(self, dropout_probability):
        self.dropout_probability = dropout_probability

    def enable_use_regression(self, use_regression):
        self.use_regression = use_regression

    def set_log_filename(self, log_filename):
        self.log_filename = log_filename

    def get_weights(self, parameters):
        current = 0

        for node in self.nodes:
            node.get_weights(current, parameters)

        for edge in self.edges:
            parameters[current] = edge.__weight
            current += 1

        for re in self.recurrent_edges:
            parameters[current] = re.__weight
            current += 1

    def set_weights(self, parameters):

        if len(parameters) != self.get_number_weights():
            log.error(f"ERROR! Trying to set weights where the RNN has {self.get_number_weights()} weights, and the parameters vector has {len(parameters)} weights!\n")
            exit(1)

        current = 0

        for node in self.nodes:
            node.set_weights(current, parameters)

        for edge in self.edges:
            edge.__weight = bound(parameters[current])
            current += 1

        for re in self.recurrent_edges:
            re.__weight = bound(parameters[current])
            current += 1

    def get_number_inputs(self):
        number_inputs = 0
        for node in self.nodes:
            if node.get_layer_type() == 0:
                number_inputs += 1
        return number_inputs

    def get_number_outputs(self):
        number_outputs = 0
        for node in self.nodes:
            if node.get_layer_type() == 2:
                number_outputs += 1
        return number_outputs

    def get_number_weights(self):
        number_weights = 0

        for node in self.nodes:
            number_weights += node.get_number_weights()

        for _ in self.edges:
            number_weights += 1

        for _ in self.recurrent_edges:
            number_weights += 1

        return number_weights

    def get_avg_edge_weight(self):
        weights = 0

        for i in range(len(self.edges)):
            edge = self.edges[i]
            if edge.enabled:
                if edge.__weight > 10:
                    log.error(f"ERROR: edge {i} has weight {edge.__weight}\n")
                weights += edge.__weight

        for i in range(len(self.recurrent_edges)):
            edge = self.recurrent_edges[i]
            if edge.enabled:
                if edge.__weight > 10:
                    log.error(f"ERROR: edge {i} has weight {edge.__weight}\n")
                weights += edge.__weight

        N = len(self.edges) + len(self.recurrent_edges)
        avg_weight = weights / N
        return avg_weight

    def initialize_randomly(self):
        log.trace(f"initializing genome {self.generation_id} of group {self.group_id} randomly")
        num_weights = self.get_number_weights()
        self.initial_parameters = [0] * num_weights

        if self.weight_initialize == WeightType.RANDOM:
            for p in self.initial_parameters:
                p = np.random.uniform(0, random(0, self.seed), 1)
            self.set_weights(self.initial_parameters)

        elif self.weight_initialize == WeightType.XAVIER:
            for node in self.nodes:
                self.initialize_xavier(node)
            self.get_weights(self.initial_parameters)

        elif self.weight_initialize == WeightType.KAIMING:
            for node in self.nodes:
                self.initialize_kaiming(node)
            self.get_weights(self.initial_parameters)

        else:
            log.error(f"ERROR: trying to initialize a genome randomly with unknown weight initialization strategy: {self.weight_initialize}\n")
            exit(1)

        self.set_best_parameters(self.initial_parameters)
        self.set_weights(self.initial_parameters)

    def initialize_xavier(self, n):
        input_edges = []
        input_recurrent_edges = []

        self.get_input_edges(n._innovation_number, input_edges, input_recurrent_edges)
        fan_in = len(input_edges) + len(input_recurrent_edges)
        fan_out = self.get_fan_out(n._innovation_number)

        sum = fan_in + fan_out
        if sum <= 0:
            sum = 1
        range = math.sqrt(6) / math.sqrt(sum)
        for edge in input_edges:
            edge_weight = range * self.rng_1_1
            edge = edge_weight

        for re in input_recurrent_edges:
            edge_weight = range * self.rng_1_1
            re = edge_weight

        n.initialize_xavier(self.rng_1_1, range)


    def initialize_kaiming(self, n):
        input_edges = []
        input_recurrent_edges = []
        self.get_input_edges(n._innovation_number, input_edges, input_recurrent_edges)
        fan_in = len(input_edges) + len(input_recurrent_edges)
        if fan_in <= 0:
            fan_in = 1
        range = math.sqrt(2) / math.sqrt(fan_in)
        for edge in input_edges:
            weight = range * np.random.normal(mu=0, sigma=1)
            edge.__weight = weight
        for edge in input_recurrent_edges:
            weight = range * np.random.normal(mu=0, sigma=1)
            edge.__weight = weight
        n.initialize_kaiming(range)

    def get_xavier_weight(self, output_node):
        sum = self.get_fan_in(output_node._innovation_number) +  self.get_fan_out(output_node._innovation_number)
        if sum <= 0:
            sum = 1

        range =  math.sqrt(2) / math.sqrt(sum)
        return range * self.rng_1_1

    def get_kaiming_weight(self, output_node):
        fan_in = self.get_fan_in(output_node._innovation_number)
        if fan_in <= 0:
            fan_in = 1
        range = math.sqrt(2) / math.sqrt(fan_in)
        return range * np.random.normal(mu=0, sigma=1)

    def get_random_weight(self):
        return self.rng

    def initialize_node_randomly(self, n):
        if self.mutated_component_weight == self.weight_initialize:
            if self.weight_initialize == WeightType.XAVIER:
                sum = self.get_fan_in(n._innovation_number) + self.get_fan_out(n._innovation_number)
                if sum <= 0:
                    sum = 1
                range = math.sqrt(6) / math.sqrt(sum)

                n.initialize_xavier(range)
            elif self.weight_initialize == WeightType.KAIMING:
                sum = self.get_fan_in(n._innovation_number)
                if sum <= 0:
                    sum = 1
                range = math.sqrt(2) / math.sqrt(sum)
                n.initialize_kaiming(range)
            elif self.weight_initialize == WeightType.RANDOM:
                n.initialize_uniform_random(self.rng)
            else:
                log.error(f"Weight initlialize method {self.weight_initialize} is not set correclty")
                exit(1)
    def get_input_edges(self, node_innovation, input_edges, input_recurrent_edges):
        for edge in self.edges:
            if edge.enabled:
                if edge.__output_node._innovation_number == node_innovation:
                    input_edges.append(edge)

        for edge in self.recurrent_edges:
            if edge.enabled:
                if edge.__output_node._innovation_number == node_innovation:
                    input_recurrent_edges.append(edge)

    def get_fan_in(self, node_innovation):
        fan_in = 0
        for edge in self.edges:
            if edge.enabled:
                if edge.__output_node._innovation_number == node_innovation:
                    fan_in += 1

        for edge in self.recurrent_edges:
            if edge.enabled:
                if edge.__output_node._innovation_number == node_innovation:
                    fan_in += 1
        return fan_in

    def get_fan_in(self, node_innovation):
        fan_out = 0
        for edge in self.edges:
            if edge.enabled:
                if edge.__input_node._innovation_number == node_innovation:
                    fan_out += 1

        for edge in self.recurrent_edges:
            if edge.enabled:
                if edge.__input_node._innovation_number == node_innovation:
                    fan_out += 1

        return fan_out

    def get_rnn(self):
        node_copies = []
        edge_copies = []
        recurrent_edge_copies = []

        for node in self.nodes:
            node_copies.append(node.copy())

        for edge in self.edges:
            edge_copies.append(edge.copy(node_copies))

        for edge in self.recurrent_edges:
            edge_copies.append(edge.copy(node_copies))

        return RNN(nodes=node_copies, edges=edge_copies, recurrent_edges=recurrent_edge_copies, input_parameter_names=self.input_parameter_names, output_parameter_names=self.output_parameter_names)
    
