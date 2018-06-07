import Tkinter as tk


class ColorLegend(tk.Frame):

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.color_legend = tk.LabelFrame(self, text='Color legend')
        self.color_legend.grid(columnspan=2, sticky='w')
        colors = [
            ('blue', 'Neuron'),
            ('purple', 'Input data sample'),
            ('yellow', 'Current randomly selected input sample'),
            ('red', 'Best matching unit (BMU)'),
            ('green', 'Neighbors of BMU')
        ]
        for i, (color, case) in enumerate(colors):
            color_type_label = tk.Label(
                self.color_legend, text='', bg=color, pady=3, padx=3)
            color_label = tk.Label(
                self.color_legend, text=case, pady=3, padx=3)
            color_type_label.grid(row=i, column=0, sticky='we')
            color_label.grid(row=i, column=1, sticky='w')

