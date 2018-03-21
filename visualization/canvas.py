import Tkinter as tk
import numpy as np


class SomCanvas(tk.Frame):

    def __init__(self, parent, title, width, height, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.canvas = tk.Canvas(self, height=height, width=width, bg='white')
        self.title = tk.Label(self, text=title)
        self.title.grid()
        self.canvas.grid(padx=5, pady=5)
        self.nodes = {}
        self.width = width
        self.height = height

    def draw_node(self, coords, color, name):
        x1, y1, x2, y2 = coords
        node = self.canvas.create_oval(x1, y1, x2, y2, fill=color)
        self.nodes[name] = node  # save node under given name for later use

    def draw_connection(self, x, y, neighbor_x, neighbor_y):
        self.canvas.create_line(x, y, neighbor_x, neighbor_y)

    def update_item_color(self, name, color):
        """ Change color of item with given name to specified color. """
        self.canvas.itemconfig(name, fill=color)

    def reset(self):
        """ Delete all existing items on canvas. """
        self.canvas.delete('all')

    def reset_node_colors_to_blue(self):
        """ Change all node colors to blue. """
        for node in self.nodes.values():
            self.update_item_color(node, 'blue')


class MapGrid(SomCanvas):

    def __init__(self, parent, title, width, height, *args, **kwargs):
        SomCanvas.__init__(self, parent, title, width, height, *args, **kwargs)

    def draw_nodes(self, som):
        shape = som.shape
        scale_x = self.width / shape[1]
        scale_y = self.height / shape[0]
        shift = 0.5
        for index, node in np.ndenumerate(som.map):
            vector = node.map_index
            x, y = scale_and_shift_vector(vector, scale_x, scale_y, shift)
            coords = get_node_coords(x, y, shape)
            name = vector
            self.draw_node(coords, 'blue', name)
            self.draw_connections(shape, index, x, y, scale_x, scale_y, shift)

    def draw_connections(self, shape, index, x, y, scale_x, scale_y, shift):
        neighbor_indices = get_neighbor_indices(index, shape)
        for neighbor_index in neighbor_indices:
            neighbor_x, neighbor_y = scale_and_shift_vector(
                neighbor_index, scale_x, scale_y, shift)
            self.draw_connection(x, y, neighbor_x, neighbor_y)

    def draw_u_matrix(self, u_matrix):
        shape = u_matrix.shape
        min_distance = u_matrix.min()
        max_distance = u_matrix.max()
        distance_range = max_distance - min_distance
        for i in range(shape[0]):
            for j in range(shape[1]):
                # TODO
                x = (i + 0.5) * (self.width / shape[1])
                y = (j + 0.5) * (self.height / shape[0])
                x_length = self.width / shape[1] / 2
                y_length = self.height / shape[0] / 2
                # radius = 200 / (self.shape[0] * self.shape[1])
                x1, y1 = (x - x_length), (y - y_length)
                x2, y2 = (x + x_length), (y + y_length)
                # color interpolation (99 grays in tkinter)
                distance_scaled = float(u_matrix[i, j] - min_distance) / float(distance_range)
                color_number = int(1 + (distance_scaled * 98))
                color = 'gray%d' % (100 - color_number)  # gray1 = black
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=color)


class FeatureSpaceGrid(SomCanvas):
    """ Canvas for visualizing the feature space of the som. In the feature
    space the weight vectors of nodes in the som and the weight vectors of the
    samples of the input data are drawn."""

    def __init__(self, parent, title, width, height, *args, **kwargs):
        SomCanvas.__init__(self, parent, title, width, height, *args, **kwargs)
        self.samples = {}
    
    def draw_nodes(self, som):
        shape = som.shape
        shift = 0
        scale_x = self.width
        scale_y = self.height
        for index, node in np.ndenumerate(som.map):
            vector = node.weight_vector
            x, y = scale_and_shift_vector(vector, scale_x, scale_y, shift)
            coords = get_node_coords(x, y, shape)
            name = node.map_index
            self.draw_node(coords, 'blue', name)
        self.draw_connections(shape)

    def draw_connections(self, shape):
        self.canvas.delete('line')  # delete existing drawn connections
        for map_index, node in self.nodes.items():
            x = int(map_index[0])
            y = int(map_index[1])
            coords = self.canvas.coords(node)
            old_x = (coords[0] + coords[2]) / 2
            old_y = (coords[1] + coords[3]) / 2
            neighbor_indices = get_neighbor_indices((x, y), shape)
            for neighbor_x, neighbor_y in neighbor_indices:
                # get corresponding node
                name = (neighbor_x, neighbor_y)
                coords = self.canvas.coords(self.nodes[name])
                new_x = (coords[0] + coords[2]) / 2
                new_y = (coords[1] + coords[3]) / 2
                self.canvas.create_line(old_x, old_y, new_x, new_y,
                                        tags='line')

    def draw_sample(self, sample, shape):
        x1, y1, x2, y2 = get_node_coords(sample[0], sample[1], shape)
        sample = self.canvas.create_oval(x1, y1, x2, y2, fill='purple')
        self.samples[str(sample)] = sample

    def reset_sample_color_to_purple(self):
        for sample in self.samples.values():
            self.update_item_color(sample, 'purple')

    def update_node(self, name, new_weights, shape):
        x, y = scale_and_shift_vector(new_weights, self.width, self.height, 0)
        new_coords = get_node_coords(x, y, shape)
        self.canvas.coords(self.nodes[name], new_coords)

    def get_node_by_weight_vector(self, weight_vector, shape):
        x, y = scale_and_shift_vector(weight_vector, self.width, self.height, 0)
        weight_vector_coords = get_node_coords(x, y, shape)
        for node in self.canvas.find('all'):
            node_coords = tuple([int(coord)
                                 for coord in self.canvas.coords(node)])
            if node_coords == weight_vector_coords:
                return node

    def draw_selected_input_vector(self, input_vector, shape):
        input_vector = self.get_node_by_weight_vector(input_vector, shape)
        self.update_item_color(input_vector, 'yellow')


def scale_and_shift_vector(vector, scale_x, scale_y, shift):
    x = (vector[0] + shift) * scale_x
    y = (vector[1] + shift) * scale_y
    return x, y


def get_node_coords(x, y, shape):
    # TODO: Do not let radius depend on shape here
    radius = 200 / (shape[0] * shape[1])
    x1, y1 = int(x - radius), int(y - radius)
    x2, y2 = int(x + radius), int(y + radius)
    return x1, y1, x2, y2


def get_neighbor_indices(index, shape):
    """ Return all indices of direct neighbors for the given index.

    >>> get_neighbor_indices((0, 0), (2, 2))
    [(0, 1), (1, 0), (1, 1)]
    >>> get_neighbor_indices((1, 1), (3, 3))
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1), (2, 2)]

    :param index:
    :param shape:
    :return:
    """
    neighbor_indices = []
    for neighbor_i in [-1, 0, 1]:
        for neighbor_j in [-1, 0, 1]:
            row_index = index[0] + neighbor_i
            col_index = index[1] + neighbor_j
            if row_index < 0 or col_index < 0:
                continue
            if row_index >= shape[0] or col_index >= shape[1]:
                continue
            if neighbor_j == neighbor_i == 0:
                continue
            neighbor_indices.append((row_index, col_index))
    return neighbor_indices

