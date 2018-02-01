import curses
import time
import numpy as np


def euclidean_distance(vector_x, vector_y):
    """ Calculate euclidean distance between two vectors.

    >>> vector_x = np.array([0.0, 0.0, 0.0])
    >>> euclidean_distance(vector_x, np.array([0.0, 0.0, 0.0]))
    0.0
    >>> euclidean_distance(vector_x, np.array([2.0, 2.0, 1.0]))
    3.0
    >>> euclidean_distance(vector_x, np.array([1.0, 1.0, 1.0]))
    1.7320508075688772

    """
    distance = 0
    length = min(len(vector_x), len(vector_y))
    for i in range(length):
        distance += (vector_x[i] - vector_y[i]) ** 2
    return np.sqrt(distance)



class SelfOrganizingMap(object):
    """ Class for the generation of self-organizing maps

    TODO: General usage example

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


from Tkinter import Tk, Button, LabelFrame, Label, Canvas, Entry, Scale
from tkMessageBox import showinfo


class SelfOrganizingMapVisualisator:

    def __init__(self):

        # visualization
        self.canvas_width = 500
        self.canvas_height = 500

        self.master = Tk()  # the root window
        self.master.title('Visualization of SOM Algorithm')

        # entries
        self.num_rows_label = Label(self.master, text='Number of rows: ')
        self.num_rows_label.grid(row=1, column=0, sticky='w')
        self.num_rows_entry = Entry(self.master, width=10)
        self.num_rows_entry.grid(row=1, column=0, sticky='e')
        self.num_rows_entry.insert(0, '5')

        self.num_cols_label = Label(self.master, text='Number of columns: ')
        self.num_cols_label.grid(row=2, column=0, sticky='w')
        self.num_cols_entry = Entry(self.master, width=10)
        self.num_cols_entry.grid(row=2, column=0, sticky='e')
        self.num_cols_entry.insert(0, '5')

        self.iteration_limit_label = Label(self.master, text='Iteration limit:')
        self.iteration_limit_label.grid(row=3, column=0, sticky='w')
        self.iteration_limit_entry = Entry(self.master, width=10)
        self.iteration_limit_entry.insert(0, '1000')
        self.iteration_limit_entry.grid(row=3, column=0, sticky='e')

        self.learning_rate_label = Label(self.master, text='Initial learning rate:')
        self.learning_rate_label.grid(row=4, column=0, sticky='w')
        self.learning_rate_entry = Entry(self.master, width=10)
        self.learning_rate_entry.insert(0, '0.1')
        self.learning_rate_entry.grid(row=4, column=0, sticky='e')

        self.neighbor_radius_label = Label(self.master, text='Initial neighbor radius:')
        self.neighbor_radius_label.grid(row=5, column=0, sticky='w')
        self.neighbor_radius_entry = Entry(self.master, width=10)
        self.neighbor_radius_entry.insert(0, '3.0')
        self.neighbor_radius_entry.grid(row=5, column=0, sticky='e')

        self.num_samples_label = Label(self.master, text='Samples: 0')
        self.num_samples_label.grid(row=7, column=0, sticky='w')

        # init button
        self.initialize_button = Button(self.master, text='Initialize map', command=self.initialize_map)
        self.initialize_button.grid(row=9, column=0, sticky='w')
        self.reset_button = Button(self.master, text='Reset', command=self.reset)
        self.reset_button.grid(row=9, column=0, sticky='e')

        # legend for main algorithm
        self.legend = LabelFrame(self.master, text='SOM Algorithm steps')
        self.legend.grid(row=10, column=0, sticky='wesn', padx=2, pady=2)
        self.algorithm_steps = {
            0: '1. Pick a random input sample',
            1: '2. Find best matching unit (BMU)',
            2: '3. Find neighbors of BMU',
            3: '4. Update weight vectors',
            4: '5. Repeat until iteration limit is reached'
}
        self.algorithm_step_labels = []
        for num, algorithm_step in sorted(self.algorithm_steps.items()):
            label = Label(self.legend, text=algorithm_step)
            label.grid(row=num, column=0, sticky='w')
            self.algorithm_step_labels.append(label)

        # display variables
        self.variables = LabelFrame(self.master, text='Variables')
        self.variables.grid(row=12, column=0, sticky='wesn')
        self.iteration_step_label = Label(self.variables)
        self.iteration_step_label.grid(row=0, column=0, sticky='w')
        self.learning_rate_label = Label(self.variables)
        self.learning_rate_label.grid(row=1, column=0, sticky='w')
        self.neighbor_radius_label = Label(self.variables)
        self.neighbor_radius_label.grid(row=2, column=0, sticky='w')

        # algorithm buttons
        self.play_button = Button(self.master, text='Play', command=self.play, state='disabled')
        self.play_button.grid(row=13, column=0, sticky='w')
        self.step_button = Button(self.master, text='Step', command=self.execute_algorithm_step, state='disabled')
        self.step_button.grid(row=13, column=0, sticky='e')
        self.pause_button = Button(self.master, text='Pause', command=self.pause, state='disabled')
        self.pause_button.grid(row=13, column=0, sticky='')

        # speed slider
        self.speed_label = Label(self.master, text='Speed (left = slow, right = fast)')
        self.speed_label.grid(row=14, column=0, sticky='w')
        self.speed_slider = Scale(self.master, from_=1000, to=1, orient='horizontal', showvalue=0)
        self.speed_slider.set(500)
        self.speed_slider.config(state='disabled')
        self.speed_slider.grid(row=14, column=0, sticky='wesn')

        # color legend
        self.color_legend = LabelFrame(self.master, text='Color legend')
        self.color_legend.grid(row=17, column=0, columnspan=2, sticky='w')
        colors = [
            ('blue', 'Node'),
            ('purple', 'Input data sample'),
            ('yellow', 'Current randomly selected input sample'),
            ('red', 'Best matching unit (BMU)'),
            ('green', 'Neighbors of BMU')
        ]
        for i, (color, case) in enumerate(colors):
            color_type_label = Label(self.color_legend, text='', bg=color, pady=3, padx=3)
            color_type_label.grid(row=i, column=0, sticky='we')
            color_label = Label(self.color_legend, text=case, pady=3, padx=3)
            color_label.grid(row=i, column=1, sticky='w')

        # map
        self.map_grid = Canvas(self.master, width=self.canvas_width, height=self.canvas_height, bg='white')
        self.map_grid.grid(row=1, column=3, rowspan=20, padx=5, pady=5, sticky='wens')
        self.map_grid_title = Label(self.master, text='Node grid')
        self.map_grid_title.grid(row=0, column=3, sticky='wesn')

        # visualisation of weight vectors of nodes in som and input vectors
        self.map = Canvas(self.master, width=self.canvas_width, height=self.canvas_height, bg='white')
        # track mouse clicks
        self.map.bind('<Button-1>', self.mouse_click_action)
        self.map.grid(row=1, column=4, rowspan=20, padx=5, pady=5, sticky='wens')
        self.map_title = Label(self.master, text='Node weight vectors and input data (samples) weight vectors')
        self.map_title.grid(row=0, column=4, sticky='wesn')

        # variables
        self.running = False

        # som
        self.algorithm_step = 0
        self.samples = []
        self.data = None
        self.som = None
        self.shape = None
        self.nodes = {}
        self.nodes_grid = {}
        self.selected_input_vector_id = None  # name on canvas
        self.bmu = None  # name on canvas
        self.neighbor_ids = None  # names of neighbor nodes of bmu on canvas

        # main loop of Tkinter GUI
        self.master.mainloop()

    def init_som(self):
        x = int(self.num_rows_entry.get())
        y = int(self.num_cols_entry.get())
        if x < 0 or y < 0:
            showinfo('', 'Rows and Columns must be positive!')
            return None
        self.shape = (x, y)
        iteration_limit = int(self.iteration_limit_entry.get())
        if iteration_limit <= 1:
            showinfo('', 'Iteration limit must be minimum 2')
            return None
        init_learning_rate = float(self.learning_rate_entry.get())
        if init_learning_rate > 1 or init_learning_rate < 0:
            showinfo('', 'Initial learning rate must be between 0 and 1!')
            return None
        init_neighbor_radius = float(self.neighbor_radius_entry.get())
        if init_neighbor_radius < 0:
            showinfo('', 'Initial neighbor radius must be positive!')
            return None
        # random data dummy for initialization (to enable drawing of samples)
        # data will be updated later within execute_algorithm
        data = np.random.random((1, 2))
        self.som = SelfOrganizingMap(self.shape, data, iteration_limit,
                                     init_learning_rate, init_neighbor_radius)
        self.som.initialize_map()

    def initialize_map(self):
        self.init_som()
        self.draw_nodes()  # left canvas
        self.draw_weights()  # right canvas
        self.num_rows_entry.config(state='disabled')
        self.num_cols_entry.config(state='disabled')
        self.iteration_limit_entry.config(state='disabled')
        self.initialize_button.config(state='disabled')
        self.neighbor_radius_entry.config(state='disabled')
        self.learning_rate_entry.config(state='disabled')
        self.play_button.config(state='normal')
        self.pause_button.config(state='normal')
        self.step_button.config(state='normal')
        self.speed_slider.config(state='normal')

    def draw_weights(self, connections=True):
        # draw randomly initialized node weight vectors
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                node = self.som.map[i, j]
                self.draw_node(node)
        if connections:
            self.draw_connections()

    def draw_connections(self):
        self.map.delete('line')  # delete existing drawn connections
        for name, node in self.nodes.items():
            map_index = name.split('-')[1:]
            x = int(map_index[0])
            y = int(map_index[1])
            coords = self.map.coords(node)
            old_x = (coords[0] + coords[2]) / 2
            old_y = (coords[1] + coords[3]) / 2
            for i in [-1, 0, 1]:
                for j in [-1, 0, 1]:
                    neighbor_x = (x + i)
                    neighbor_y = (y + j)
                    if neighbor_x < 0 or neighbor_y < 0:
                        continue
                    if neighbor_x >= self.shape[0] or neighbor_y >= \
                            self.shape[1]:
                        continue
                    # get corresponding node
                    name = 'node-%s-%s' % (neighbor_x, neighbor_y)
                    coords = self.map.coords(self.nodes[name])
                    new_x = (coords[0] + coords[2]) / 2
                    new_y = (coords[1] + coords[3]) / 2
                    self.map.create_line(old_x, old_y, new_x, new_y,
                                         tags='line')

    def reset(self):
        self.running = False
        self.map.delete('all')
        self.map_grid.delete('all')
        self.nodes = {}
        self.nodes_grid = {}
        self.samples = []
        step = self.algorithm_step % 5
        self.update_algorithm_step_labels(step, reset=True)
        self.algorithm_step = 0
        self.update_variable_labels()
        self.num_samples_label['text'] = 'Samples: 0'
        self.num_cols_entry.config(state='normal')
        self.num_rows_entry.config(state='normal')
        self.initialize_button.config(state='normal')
        self.iteration_limit_entry.config(state='normal')
        self.learning_rate_entry.config(state='normal')
        self.neighbor_radius_entry.config(state='normal')
        self.play_button.config(state='disabled')
        self.pause_button.config(state='disabled')
        self.step_button.config(state='disabled')
        self.speed_slider.config(state='disabled')

    def play(self):
        self.running = True
        self.run()

    def run(self):
        if self.running:
            self.execute_algorithm_step()
        self.master.after(int(self.speed_slider.get()), self.run)

    def pause(self):
        self.running = False

    def execute_algorithm_step(self):
        if len(self.samples) <= 1:
            self.running = False
            showinfo('', 'Please draw at least two training samples by '
                         'clicking somewhere on the right canvas with '
                         'the mouse!')
            return None
        self.check_for_new_samples()
        step = self.algorithm_step % 5
        self.update_algorithm_step_labels(step)
        self.update_variable_labels()
        if step == 0:
            self.pick_and_draw_random_input_sample()
        elif step == 1:
            self.find_and_draw_bmu()
        elif step == 2:
            self.find_and_draw_neighbors()
        elif step == 3:
            self.update_and_draw_weight_vectors()
        elif step == 4:
            self.prepare_next_iteration()
            # manually update soms iteration step because this would have been
            # done within the train method
            self.som.current_iteration_step += 1
        self.algorithm_step += 1

    def draw_nodes(self):
        for i in range(self.shape[0]):
            for j in range(self.shape[1]):
                x = (i + 0.5) * (self.canvas_width / self.shape[1])
                y = (j + 0.5) * (self.canvas_height / self.shape[0])
                radius = 200 / (self.shape[0] * self.shape[1])
                x1, y1 = (x - radius), (y - radius)
                x2, y2 = (x + radius), (y + radius)
                name = str((i, j))
                self.nodes_grid[name] = self.map_grid.create_oval(x1, y1, x2, y2, fill='blue')
                # draw connections
                for neighbor_i in [-1, 0, 1]:
                    for neighbor_j in [-1, 0, 1]:
                        row_index = i + neighbor_i
                        col_index = j + neighbor_j
                        if row_index < 0 or col_index < 0:
                            continue
                        if row_index >= self.shape[0] or col_index >= self.shape[1]:
                            continue
                        neighbor_x = (row_index + 0.5) * (self.canvas_width / self.shape[1])
                        neighbor_y = (col_index + 0.5) * (self.canvas_width / self.shape[1])
                        self.map_grid.create_line(x, y, neighbor_x, neighbor_y)

    def mouse_click_action(self, event):
        """ If map is initialized, enable drawing of samples by mouseclick. """
        if self.initialize_button['state'] == 'disabled':
            input_vector = (event.x, event.y)
            self.samples.append(input_vector)
            self.draw_input_vector(input_vector)
            self.num_samples_label['text'] = 'Samples: %s' % len(self.samples)

    def get_input_data_from_samples(self):
        """ Set the som's input data to all currently drawn input samples. """
        data = np.empty((len(self.samples), 2))
        for i, sample in enumerate(self.samples):
            x = sample[0] / float(self.canvas_height)
            y = sample[1] / float(self.canvas_width)
            data[i, :] = np.array([x, y])
        return data

    def update_algorithm_step_labels(self, step, reset=False):
        """ Mark only the current algorithm step with white background. """
        if not reset:
            current_label = self.algorithm_step_labels[step]
            current_label.config(bg='white')
        last_label = self.algorithm_step_labels[step - 1]
        last_label.config(bg=self.master.cget('bg'))  # switch back to default

    def get_canvas_item_name_by_map_index(self, map_index):
        return 'node-%s-%s' % (map_index[0], map_index[1])

    def pick_and_draw_random_input_sample(self):
        self.som.pick_random_input_vector()
        input_vector = self.get_map_id_from_weight_vector(self.som.input_vector)
        self.selected_input_vector_id = input_vector
        self.update_canvas_item_color(self.selected_input_vector_id, 'yellow')

    def find_and_draw_bmu(self):
        self.som.find_bmu()
        bmu_index = self.som.bmu.map_index
        name = self.get_canvas_item_name_by_map_index(bmu_index)
        self.bmu = self.nodes[name]
        self.update_canvas_item_color(self.bmu, 'red')
        self.map_grid.itemconfig(self.nodes_grid[str(bmu_index)], fill='red')

    def find_and_draw_neighbors(self):
        self.neighbor_ids = []
        for neighbor in self.som.find_neighbors():
            name = self.get_canvas_item_name_by_map_index(neighbor.map_index)
            self.neighbor_ids.append(name)
            if neighbor.map_index != self.som.bmu.map_index:
                self.update_canvas_item_color(self.nodes[name], 'green')
                self.map_grid.itemconfig(self.nodes_grid[str(neighbor.map_index)],
                                         fill='green')

    def update_and_draw_weight_vectors(self):
        self.som.update_map()
        for neighbor in self.neighbor_ids:
            map_index = tuple([int(x) for x in neighbor.split('-')[1:]])
            new_weights = self.som.map[map_index].weight_vector
            new_coords = self.get_coords_from_vector(new_weights)
            self.map.coords(self.nodes[neighbor], new_coords)
        self.draw_connections()

    def prepare_next_iteration(self):
        self.update_canvas_item_color(self.selected_input_vector_id, 'purple')
        self.update_canvas_item_color(self.bmu, 'blue')
        for neighbor in self.neighbor_ids:
            self.update_canvas_item_color(self.nodes[neighbor], 'blue')
        for node in self.nodes_grid.values():
            self.map_grid.itemconfig(node, fill='blue')
        if self.som.current_iteration_step == self.som.iteration_limit:
            self.play_button.config(state='disabled')
            self.pause_button.config(state='disabled')
            self.step_button.config(state='disabled')
            self.running = False

    def update_canvas_item_color(self, name, color):
        """ Change color of item with given name to specified color. """
        self.map.itemconfig(name, fill=color)

    def get_map_id_from_weight_vector(self, weight_vector):
        weight_vector_coords = self.get_coords_from_vector(weight_vector)
        for item in self.map.find('all'):
            coords = tuple([int(coord) for coord in self.map.coords(item)])
            if coords == weight_vector_coords:
                return item

    def update_variable_labels(self):
        self.iteration_limit_label['text'] = \
            'Iteration limit: %s' \
            % self.som.iteration_limit
        self.iteration_step_label['text'] = \
            'Current iteration step: %s' \
            % self.som.current_iteration_step
        self.neighbor_radius_label['text'] = \
            'Current neighbor radius: %.6f' \
            % self.som.get_neighborhood_radius()
        self.learning_rate_label['text'] = \
            'Current learning rate: %.6f' \
            % self.som.get_learning_rate()

    def check_for_new_samples(self):
        """ Check if new samples have been drawn since last training cycle. """
        if len(self.samples) > self.som.data.shape[0]:
            self.som.data = self.get_input_data_from_samples()

    def get_coords_from_vector(self, vector, scale=True):
        radius = 200 / (self.shape[0] * self.shape[1])
        x, y, = vector[0], vector[1]
        if scale:
            x = x * self.canvas_width
            y = y * self.canvas_height
        x1, y1 = int(x - radius), int(y - radius)
        x2, y2 = int(x + radius), int(y + radius)
        return x1, y1, x2, y2

    def draw_node(self, node):
        """ Draw two-dimensional weight vector as small oval. """
        x1, y1, x2, y2 = self.get_coords_from_vector(node.weight_vector)
        name = self.get_canvas_item_name_by_map_index(node.map_index)
        self.nodes[name] = self.map.create_oval(x1, y1, x2, y2, fill='blue')

    def draw_input_vector(self, input_vector):
        """ Draw input vector on right canvas. """
        x1, y1, x2, y2 = self.get_coords_from_vector(input_vector, scale=False)
        self.map.create_oval(x1, y1, x2, y2, fill='purple')


if __name__ == '__main__':
    sv = SelfOrganizingMapVisualisator()
