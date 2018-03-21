import Tkinter as tk
import tkMessageBox
from parameters import Parameters
from speedslider import SpeedSlider
from algorithmvariables import AlgorithmVariables
from colorlegend import ColorLegend
from canvas import MapGrid, FeatureSpaceGrid
from algorithmstepslegend import AlgorithmStepsLegend


class SomVisualization(tk.Tk):

    def __init__(self, controller, canvas_width=500, canvas_height=500,
                 *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)
        self.title('Visualization of SOM Algorithm')

        self.controller = controller

        self.parameters = Parameters(self)

        self.num_samples = tk.Label(self, text='Samples: 0')

        self.initialize_button = tk.Button(self, text='Initialize map', command=self.initialize)
        self.reset_button = tk.Button(self, text='Reset', command=self.reset)
        self.u_matrix_button = tk.Button(self.master, text='Draw U-Matrix', command=self.draw_u_matrix, state='disabled')

        self.play_button = tk.Button(self.master, text='Play', command=self.play, state='disabled')
        self.step_button = tk.Button(self.master, text='Step', command=self.step, state='disabled')
        self.pause_button = tk.Button(self.master, text='Pause', command=self.pause, state='disabled')

        self.speed_slider = SpeedSlider(self)

        self.algorithm_steps_legend = AlgorithmStepsLegend(self)
        self.algorithm_variables = AlgorithmVariables(self)
        self.algorithm_variables.init_labels(self.parameters.learning_rate(),
                                             self.parameters.neighbor_radius())
        self.color_legend = ColorLegend(self)

        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.map_grid = MapGrid(self, title='Node Grid', width=self.canvas_width, height=self.canvas_height)
        self.feature_space_grid = FeatureSpaceGrid(self, title='lala', width=self.canvas_width, height=self.canvas_height)
        self.feature_space_grid.canvas.bind('<Button-1>', self.mouse_click_action)

        # placement of items in main window
        self.parameters.grid(row=0)
        self.num_samples.grid(row=1, sticky='w', padx=10, pady=10)
        self.initialize_button.grid(row=2, sticky='w')
        self.reset_button.grid(row=2, sticky='e')
        self.play_button.grid(row=3, sticky='w')
        self.step_button.grid(row=3, sticky='e')
        self.pause_button.grid(row=3, sticky='')
        self.speed_slider.grid(padx=10, pady=10)
        self.u_matrix_button.grid()
        self.algorithm_steps_legend.grid(padx=10, pady=10)
        self.algorithm_variables.grid(sticky='wesn', padx=10, pady=10)
        self.color_legend.grid()
        self.map_grid.grid(row=0, column=1, rowspan=9)
        self.feature_space_grid.grid(row=0, column=2, rowspan=9)

    def initialize(self):
        self.parameters.config(state='disabled')
        self.initialize_button.config(state='disabled')
        self.play_button.config(state='normal')
        self.pause_button.config(state='normal')
        self.step_button.config(state='normal')
        self.speed_slider.config(state='normal')
        self.controller.initialize()

    def reset(self):
        self.feature_space_grid.reset()
        self.map_grid.reset()
        self.num_samples['text'] = 'Samples: 0'
        self.parameters.config(state='normal')
        self.initialize_button.config(state='normal')
        self.play_button.config(state='disabled')
        self.pause_button.config(state='disabled')
        self.step_button.config(state='disabled')
        self.speed_slider.config(state='disabled')
        self.u_matrix_button.config(state='disabled')
        self.algorithm_variables.init_labels(self.parameters.learning_rate(),
                                             self.parameters.neighbor_radius())
        self.controller.reset()

    def iteration_limit_reached_state(self):
        self.play_button.config(state='disabled')
        self.pause_button.config(state='disabled')
        self.step_button.config(state='disabled')
        self.u_matrix_button.config(state='normal')

    def play(self):
        if len(self.controller.samples) <= 1:
            tkMessageBox.showinfo('', 'Please draw at least two training '
                                      'samples by clicking somewhere on the '
                                      'right canvas with the mouse!')
            return None
        self.controller.play()

    def step(self):
        if len(self.controller.samples) <= 1:
            tkMessageBox.showinfo('', 'Please draw at least two training '
                                      'samples by clicking somewhere on the '
                                      'right canvas with the mouse!')
            return None
        self.controller.execute_algorithm_step()

    def pause(self):
        self.controller.pause()

    def draw_u_matrix(self):
        self.controller.draw_u_matrix()

    def mouse_click_action(self, event):
        """ If map is initialized, enable drawing of samples by mouseclick. """
        if self.initialize_button['state'] == 'disabled':
            sample = (event.x, event.y)
            self.controller.add_sample(sample)

