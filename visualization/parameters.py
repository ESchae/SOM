import Tkinter as tk


class Parameters(tk.Frame):

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self._num_rows = LabelEntry(self, 'Number of rows:', '5')
        self._num_cols = LabelEntry(self, 'Number of columns:', '5')
        self._iteration_limit = LabelEntry(self, 'Iteration limit:', '1000')
        self._learning_rate = LabelEntry(self, 'Initial learning rate:', '0.1')
        self._neighbor_radius = LabelEntry(self, 'Initial neighbor radius:', '3.0')

        self._num_rows.grid(row=0, column=0)
        self._num_cols.grid(row=1, column=0)
        self._iteration_limit.grid(row=2, column=0)
        self._learning_rate.grid(row=3, column=0)
        self._neighbor_radius.grid(row=4, column=0)

    def num_rows(self):
        return int(self._num_rows.entry.get())

    def num_cols(self):
        return int(self._num_cols.entry.get())

    def iteration_limit(self):
        return int(self._iteration_limit.entry.get())

    def learning_rate(self):
        return float(self._learning_rate.entry.get())

    def neighbor_radius(self):
        return float(self._neighbor_radius.entry.get())

    def valid(self):
        """ Return true if given parameters are valid, else notify via
         tkMessagebox and return False. """
        if (self.valid_num_rows_and_cols() and self.valid_iteration_limit() and
                self.valid_learning_rate() and self.valid_neighbor_radius()):
            return True
        else:
            return False

    def valid_num_rows_and_cols(self):
        if self.num_rows() < 0 or self.num_cols() < 0:
            tkMessageBox.showinfo(
                '', 'Rows and Columns must be positive!')
            return False
        else:
            return True

    def valid_iteration_limit(self):
        if self.iteration_limit() <= 1:
            tkMessageBox.showinfo(
                '', 'Iteration limit must be minimum 2')
            return False
        else:
            return True

    def valid_learning_rate(self):
        if self.learning_rate() > 1 or self.learning_rate() < 0:
            tkMessageBox.showinfo(
                '', 'Initial learning rate must be between 0 and 1!')
            return False
        else:
            return True

    def valid_neighbor_radius(self):
        if self.neighbor_radius() < 0:
            tkMessageBox.showinfo(
                '', 'Initial neighbor radius must be positive!')
            return False
        else:
            return True

    def config(self, *args, **kwargs):
        """ Set config for all parameter entries at once. """
        self._num_rows.entry.config(*args, **kwargs)
        self._num_cols.entry.config(*args, **kwargs)
        self._iteration_limit.entry.config(*args, **kwargs)
        self._neighbor_radius.entry.config(*args, **kwargs)
        self._learning_rate.entry.config(*args, **kwargs)


class LabelEntry(tk.Frame):

    def __init__(self, parent, label, default, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.label = tk.Label(self, text=label, width=30, anchor='w')
        self.entry = tk.Entry(self, width=10)
        self.entry.insert(0, default)
        self.label.grid(row=0, sticky='w')
        self.entry.grid(row=0, sticky='e')


