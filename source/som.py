import numpy as np
from node import Node
from utils import euclidean_distance, get_neighbor_indices


class SelfOrganizingMap(object):
    """ Class for the generation of self-organizing maps

    on data: rows = input_vectors (the samples)
             cols = features

    """

    def __init__(self, shape, data, iteration_limit=5, init_learning_rate=0.1,
                 init_neighborhood_radius=None):
        """

        >>> data = np.random.random((3, 10))
        >>> som = SelfOrganizingMap(shape=(2,3), data=data)
        >>> som.map
        array([[0, 0, 0],
               [0, 0, 0]], dtype=object)
        >>> som.init_neighborhood_radius
        2.0
        >>> som.current_iteration_step
        0

        :param shape:
        :param iteration_limit:
        :param init_learning_rate:
        """
        self.shape = shape  # dimension of the map
        self.data = data
        self.map = np.zeros(shape, dtype=object)
        self.iteration_limit = iteration_limit
        self.current_iteration_step = 0
        self.init_learning_rate = init_learning_rate  # alpha
        if init_neighborhood_radius is None:  # sigma
            self.init_neighborhood_radius = np.ceil(max(shape) / 2) + 1
        else:
            self.init_neighborhood_radius = init_neighborhood_radius
        # constant used for calculation of current neighborhood radius
        self.radius_decay = (
                iteration_limit / np.log(self.init_neighborhood_radius))
        self.input_vector = None  # current randomly selected input vector
        self.bmu = None

    def train(self):
        """ Main algorithm of the self-organizing map.

        >>> data = np.random.random((3, 10))
        >>> som = SelfOrganizingMap(shape=(2,3), data=data, iteration_limit=6)
        >>> som.train()
        >>> som.current_iteration_step
        6

        :param data:
        :return:
        """
        self.initialize_map()
        while self.current_iteration_step < self.iteration_limit:
            self.pick_random_input_vector()
            self.find_bmu()
            self.update_map()
            self.current_iteration_step += 1
            # TODO: Maybe change this again?

    def pick_random_input_vector(self):
        """ Randomly select an input vector (= a row) from given data.

        # data set with two input vectors having four feature dimensions each
        >>> data = np.array([[1.1, 1.2, 1.3, 1.4], [2.1, 2.2, 2.3, 2.4]])
        >>> som = SelfOrganizingMap(shape=(2, 3), data=data)
        >>> np.random.seed(0)
        >>> som.pick_random_input_vector()
        >>> som.input_vector
        array([ 1.1,  1.2,  1.3,  1.4])
        >>> np.random.seed(1)
        >>> som.pick_random_input_vector()
        >>> som.input_vector
        array([ 2.1,  2.2,  2.3,  2.4])

        :param data:
        :return: input_vector
        """
        num_input_vectors = self.data.shape[0]  # rows are the input_vectors
        random_row_index = np.random.randint(0, num_input_vectors)
        self.input_vector = self.data[random_row_index, :]

    def initialize_map(self):
        """ Initialize a Neuron for each node with randomized weight vector.

        >>> data = np.random.random((3, 10))
        >>> som = SelfOrganizingMap(shape=(2,3), data=data)
        >>> som.initialize_map()
        >>> som.map  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
        array([[<som.Node object ...>, <som.Node ...>, <som.Node ...],
               [<som.Node object ...>, <som.Node ...>, <som.Node ...>]],
               dtype=object)
        >>> neuron = som.map[0,0]
        >>> neuron.map_index
        (0, 0)
        >>> neuron.weight_vector  # doctest: +ELLIPSIS
        array([ ...,  ...,  ...,  ...,  ...])

        :return: None
        """
        data_dimensionality = self.data.shape[1]  # columns are the features
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                # fill weight vector with random floats between 0 and 1
                weight_vector = np.random.random((data_dimensionality,))
                map_index = (i, j)
                self.map[i,j] = Node(weight_vector, map_index)

    def get_neighborhood_radius(self):
        """

        >>> data = np.random.random((3, 10))
        >>> som = SelfOrganizingMap(shape=(2,3), data=data)
        >>> som.get_neighborhood_radius()
        2.0
        >>> som.train()
        >>> som.get_neighborhood_radius()
        1.0

        :return:
        """
        decay = np.exp(- self.current_iteration_step / self.radius_decay)
        return self.init_neighborhood_radius * decay

    def get_learning_rate(self):
        # exponential decay
        decay = np.exp(- self.current_iteration_step / self.iteration_limit)
        return self.init_learning_rate * decay

    def find_bmu(self):
        """

        >>> data = np.random.random((3, 10))
        >>> som = SelfOrganizingMap(shape=(1, 2), data=data)
        >>> n1 = Node(np.array([0, 0, 0]), (0, 0))
        >>> n2 = Node(np.array([1, 1, 1]), (0, 1))
        >>> map = np.array([[n1, n2]])
        >>> som.map = map
        >>> som.input_vector = np.array([0, 0, 0])
        >>> som.find_bmu()
        >>> som.bmu  # doctest: +ELLIPSIS
        <som.Node object at ...>
        >>> som.bmu.map_index
        (0, 0)

        :param input_vector:
        :return:
        """
        current_min_distance = np.inf
        current_bmu = None
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                node = self.map[i, j]
                distance = node.calculate_distance(self.input_vector)
                if distance < current_min_distance:
                    current_min_distance = distance
                    current_bmu = node
        self.bmu = current_bmu

    def update_map(self):
        """ Update weight vectors of BMU and its neighbors.

        :param input_vector:
        :param bmu_index:
        :return: None
        """
        current_learning_rate = self.get_learning_rate()
        neighbors = self.find_neighbors()
        for neighbor in neighbors:
            influence = self.neighbor_influence(neighbor)
            neighbor.update(self.input_vector, current_learning_rate, influence)

    def find_neighbors(self):
        neighbors = []
        current_neighborhood_radius = self.get_neighborhood_radius()
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                node = self.map[i, j]
                distance = euclidean_distance(node.map_index, self.bmu.map_index)
                if distance <= current_neighborhood_radius:
                    neighbors.append(node)
        return neighbors

    def neighbor_influence(self, neighbor):
        """ With increasing distance the influence ... TODO """
        current_neighborhood_radius = self.get_neighborhood_radius()
        distance = euclidean_distance(neighbor.map_index, self.bmu.map_index)
        return np.exp(- distance / (2 * current_neighborhood_radius))

    def get_u_matrix(self):
        """ Return the U-Matrix of the som's map. The U-Matrix displays for
        every node the average distance between its weight vector and
        the weight vector of its neighbors.

        This method is intended to be used after training the som.

        >>> np.random.seed(1)
        >>> data = np.random.random((3, 10))
        >>> som = SelfOrganizingMap(shape=(2, 2), data=data)
        >>> som.train()
        >>> som.get_u_matrix()
        array([[ 1.14838718,  1.25679096],
               [ 1.04558795,  1.02166323]])

        """
        u_matrix = np.zeros(self.shape)
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                weight_vector = self.map[i, j].weight_vector
                neighbor_distances = 0
                neighbor_indices = get_neighbor_indices((i, j), self.shape)
                for neighbor_index in neighbor_indices:
                    neighbor_weights = self.map[neighbor_index].weight_vector
                    neighbor_distances += euclidean_distance(
                        weight_vector, neighbor_weights)
                u_matrix[i, j] = neighbor_distances / len(neighbor_indices)
        return u_matrix

