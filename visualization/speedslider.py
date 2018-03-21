import Tkinter as tk


class SpeedSlider(tk.Frame):

    def __init__(self, parent, *args, **kwargs):
        tk.Frame.__init__(self, parent, *args, **kwargs)
        self.speed_label = tk.Label(
            self, text='Speed (left = slow, right = fast)')
        self.speed_slider = tk.Scale(
            self, from_=1000, to=1, orient='horizontal', showvalue=0,
            state='disabled', length=200)
        self.speed_slider.set(500)
        self.speed_label.grid(sticky='w')
        self.speed_slider.grid(sticky='')

    def config(self, *args, **kwargs):
        """ Set config of the speed slider. """
        self.speed_slider.config(*args, **kwargs)

    def current_speed(self):
        """ Get current selected iteration speed of slider. """
        return int(self.speed_slider.get())

