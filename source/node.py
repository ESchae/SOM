from utils import euclidean_distance


class Node(object):
    """ General Neuron class, used for each node of a self-organizing map.
    """

    def __init__(self, weight_vector, map_index):
        self.weight_vector = weight_vector
        self.map_index = map_index

    def update(self, input_vector, learning_rate, influence):
        """ Move the neuron's weight vector closer to the input vector.

        >>> weight_vector = np.array([1.0, 1.0, 1.0])
        >>> map_index = (0.0, 0.0)
        >>> n = Node(weight_vector, map_index)
        >>> input_vector = np.array([2.0, 2.0, 2.0])
        >>> n.update(input_vector, 1, 0.5)
        >>> n.weight_vector
        array([ 1.5,  1.5,  1.5])

        :param input_vector:
        :param learning_rate:
        :param influence:
        :return:
        """
        delta = input_vector - self.weight_vector
        update_value = learning_rate * influence * delta
        self.weight_vector = self.weight_vector + update_value

    def calculate_distance(self, input_vector):
        """ Calculate distance between neuron's weight vector and input vector.

        >>> weight_vector = np.array([0, 0, 0])
        >>> map_index = (0, 0)
        >>> n = Node(weight_vector, map_index)
        >>> input_vector = np.array([2.0, 2.0, 1.0])
        >>> n.calculate_distance(input_vector)
        3.0

        :param input_vector:
        :return: distance
        """
        return euclidean_distance(input_vector, self.weight_vector)

