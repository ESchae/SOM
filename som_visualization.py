from visualization.visualization import SomVisualization
from source.som import SelfOrganizingMap
import numpy as np


class SomController(object):

    def __init__(self):
        self.gui = SomVisualization(self)
        self.som = None
        self.algorithm_step = 0
        self.samples = []
        self.shape = None
        self.current_neighbors = []  # map indices of current neighbors
        self.running = False

    def initialize(self):
        self.initialize_som()
        self.gui.map_grid.draw_nodes(self.som)
        self.gui.feature_space_grid.draw_nodes(self.som)

    def reset(self):
        self.running = False
        self.samples = []
        step = self.algorithm_step % 5
        self.gui.algorithm_steps_legend.set_current_step(step, reset=True)
        self.algorithm_step = 0

    def play(self):
        self.running = True
        self.run()

    def run(self):
        if self.running:
            self.execute_algorithm_step()
        self.gui.after(self.gui.speed_slider.current_speed(), self.run)

    def pause(self):
        self.running = False

    def initialize_som(self):
        if self.gui.parameters.valid():
            self.shape = (self.gui.parameters.num_rows(),
                          self.gui.parameters.num_cols())
        # random data dummy for initialization (to enable drawing of samples)
        # data will be updated later within execute_algorithm_step
        data = np.random.random((1, 2))
        self.som = SelfOrganizingMap(self.shape, data,
                                     self.gui.parameters.iteration_limit(),
                                     self.gui.parameters.learning_rate(),
                                     self.gui.parameters.neighbor_radius())
        self.som.initialize_map()

    def execute_algorithm_step(self):
        self.check_for_new_samples()
        step = self.algorithm_step % 5
        self.gui.algorithm_steps_legend.set_current_step(step)
        self.gui.algorithm_variables.update_labels(
            self.som.current_iteration_step,
            self.som.get_learning_rate(),
            self.som.get_neighborhood_radius())
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

    def check_for_new_samples(self):
        """ Check if new samples have been drawn since last training cycle. """
        if len(self.samples) > self.som.data.shape[0]:
            self.som.data = self.get_input_data_from_samples()

    def get_input_data_from_samples(self):
        """ Set the som's input data to all currently drawn input samples. """
        data = np.empty(shape=(len(self.samples), 2))
        for i, sample in enumerate(self.samples):
            x = sample[0] / float(self.gui.canvas_height)
            y = sample[1] / float(self.gui.canvas_width)
            data[i, :] = np.array([x, y])
        return data

    def add_sample(self, sample):
        self.samples.append(sample)
        self.gui.feature_space_grid.draw_sample(sample, self.shape)
        self.gui.num_samples['text'] = 'Samples: %s' % len(self.samples)

    def pick_and_draw_random_input_sample(self):
        self.som.pick_random_input_vector()
        input_vector = self.som.input_vector
        self.gui.feature_space_grid.draw_selected_input_vector(input_vector,
                                                               self.shape)

    def find_and_draw_bmu(self):
        self.som.find_bmu()
        bmu_index = self.som.bmu.map_index
        for canvas in [self.gui.map_grid, self.gui.feature_space_grid]:
            bmu_node = canvas.nodes[bmu_index]
            canvas.update_item_color(bmu_node, 'red')

    def find_and_draw_neighbors(self):
        for neighbor in self.som.find_neighbors():
            if neighbor.map_index != self.som.bmu.map_index:
                self.current_neighbors.append(neighbor.map_index)
                for canvas in [self.gui.map_grid, self.gui.feature_space_grid]:
                    neighbor_node = canvas.nodes[neighbor.map_index]
                    canvas.update_item_color(neighbor_node, 'green')

    def update_and_draw_weight_vectors(self):
        self.som.update_map()
        for neighbor in self.current_neighbors:
            new_weights = self.som.map[neighbor].weight_vector
            self.gui.feature_space_grid.update_node(
                neighbor, new_weights, self.shape)
        self.gui.feature_space_grid.draw_connections(self.shape)

    def prepare_next_iteration(self):
        self.gui.map_grid.reset_node_colors_to_blue()
        self.gui.feature_space_grid.reset_node_colors_to_blue()
        self.gui.feature_space_grid.reset_sample_color_to_purple()
        if self.som.current_iteration_step == self.som.iteration_limit:
            self.gui.iteration_limit_reached_state()
            self.running = False

    def draw_u_matrix(self):
        u_matrix = self.som.get_u_matrix()
        self.gui.map_grid.draw_u_matrix(u_matrix)


if __name__ == '__main__':
    controller = SomController()
    controller.gui.mainloop()

