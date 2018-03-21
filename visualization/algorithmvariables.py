import Tkinter as tk


class AlgorithmVariables(tk.Frame):

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.variables = tk.LabelFrame(self, text='Current Variables')
        self.iteration_step = tk.Label(self.variables, width=30, anchor='w')
        self.learning_rate = tk.Label(self.variables, width=30, anchor='w')
        self.neighbor_radius = tk.Label(self.variables, width=30, anchor='w')

        self.variables.grid()
        self.iteration_step.grid()
        self.learning_rate.grid()
        self.neighbor_radius.grid()

    def update_labels(self, iteration_step, learning_rate, neighbor_radius):
        self.iteration_step['text'] = 'Current iteration step: %s' \
                                      % iteration_step
        self.neighbor_radius['text'] = 'Current neighbor radius: %.6f' \
                                       % neighbor_radius
        self.learning_rate['text'] = 'Current learning rate: %.6f' \
                                     % learning_rate

    def init_labels(self, learning_rate, neighbor_radius, ):
        self.iteration_step['text'] = 'Current iteration step: 0'
        self.neighbor_radius['text'] = 'Current neighbor radius: %.6f' \
                                       % neighbor_radius
        self.learning_rate['text'] = 'Current learning rate: %.6f' \
                                     % learning_rate

