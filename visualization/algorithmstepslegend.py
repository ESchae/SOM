import Tkinter as tk


class AlgorithmStepsLegend(tk.Frame):

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.legend = tk.LabelFrame(self, text='SOM Algorithm steps')
        self.legend.grid(sticky='wesn', padx=2, pady=2)
        self.algorithm_steps = {
            0: '1. Pick a random input sample',
            1: '2. Find best matching unit (BMU)',
            2: '3. Find neighbors of BMU',
            3: '4. Update weight vectors',
            4: '5. Repeat until iteration limit is reached'
        }
        self.algorithm_step_labels = []
        for num, step in sorted(self.algorithm_steps.items()):
            label = tk.Label(self.legend, text=step, width=30, anchor='w')
            label.grid(sticky='w')
            self.algorithm_step_labels.append(label)

    def set_current_step(self, step, reset=False):
        """ Mark only the current algorithm step with white background. """
        if not reset:
            current_label = self.algorithm_step_labels[step]
            current_label.config(bg='white')
        last_label = self.algorithm_step_labels[step - 1]
        last_label.config(bg=self.cget('bg'))  # switch back to default


