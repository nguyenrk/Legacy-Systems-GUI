import copy
import importlib.util
import os
import pickle
import sys

import matplotlib.pyplot as plt
import saleae

from . import signal
from . import device_tests
from . import exceptions
from .utils import input_rate, display_rate, input_duration, parse_rate, prompt_yesno
from .device import FPGAPlaybackDevice
from .behavior_model import MODEL_TEMPLATE
from tkinter import *
from tkinter import ttk
from tkinter import Entry
from tkinter import scrolledtext
from subprocess import Popen, PIPE
import tkinter as tk
import tkinter.filedialog
import multiprocessing
import threading
import time
import logging
import os


EXIT_MESSAGE = 'Goodbye.'
HELP_MESSAGE = 'Available commands: {}, exit'
IO_NAME_MESSAGE = 'A signal name might be "O1" to select output 1, or "I2" to select input 2'
STATUS_MESSAGE = '''INPUTS
{}

OUTPUTS
{}

TESTS
{}
SETTINGS
{}
Behavior Model: {}'''
cmd = ' '
global clearIO
global sampleValue
class Interface:
    @staticmethod
    def build_signal_clock(sample_rate, force_sample_rate=False): #Sample rate, duty cycle, 
        while 1:
            rate = input_rate('Clock rate: ')
            duty = float(input('Duty cycle [0.50]: ') or 0.50)

            try:
                if force_sample_rate: #False
                    clock = signal.Clock(clock_rate=rate, duty_cycle=duty, sample_rate=sample_rate)
                else:
                    clock = signal.Clock(clock_rate=rate, duty_cycle=duty)
            except (exceptions.ClockTooFast, exceptions.ClockInvalidDutyCycle) as e: #Checking valide sample rates and duty cycle
                print(str(e))
                continue
            break

        return clock

    @staticmethod
    def build_signal_pulse(sample_rate, force_sample_rate=False):
        print('Pulse types: [0] High pulse  __|````|_____  [1] Low pulse   ``|____|`````')
        pulse_value = 1 - input_int('Pulse type [0]: ' )
        pre_dur = input_duration('Pre-pulse duration: ')
        dur = input_duration('Pulse duration: ')
        post_dur = input_duration('Post-pulse duration: ')

        return signal.Pulse(pulse_value, pre_dur, dur, post_dur, sample_rate=env.global_sample_rate)

    @staticmethod
    def build_signal_level(sample_rate, force_sample_rate=False):
        dur = input_duration('Duration: ')
        level = input_int('Level [0] or [1]:')
        return signal.Signal(initial_value=level, duration=dur, sample_rate=sample_rate)

    @staticmethod
    def build_signal_rs232(sample_rate, force_sample_rate=False):
        baud_rate = input_int('Baud rate: ')
        message = bytes(input('Data: '), "utf-8").decode("unicode_escape") # Parse \n as an actual newline, etc. 
        if force_sample_rate:
            return signal.RS232Signal(message, baud_rate, sample_rate=sample_rate)
        else:
            return signal.RS232Signal(message, baud_rate)

    @staticmethod
    def build_signal_can(sample_rate, force_sample_rate=False):
        baud_rate = input_int('Baud rate: ')
        message = bytes(input('Data: '), "utf-8").decode("unicode_escape") # Parse \n as an actual newline, etc. 
        if force_sample_rate:
            return signal.CANSignal(message, baud_rate, sample_rate=sample_rate)
        else:
            return signal.CANSignal(message, baud_rate)

    # TODO: implement "bitstream" mode that takes a series of 0's and 1's from the user, to be directly used as samples
    #@staticmethod
    #def build_signal_bitstream(env):

    @classmethod
    def build_signal(cls, sample_rate, force_sample_rate=False): #List of signals constructed from signal file
        signal_builders = {
            'clock': cls.build_signal_clock,
            'pulse': cls.build_signal_pulse,
            'level': cls.build_signal_level,
            'rs232': cls.build_signal_rs232,
            'can': cls.build_signal_can
        }

        _, builder_name = cls.menu_choice(list(signal_builders.keys()), prompt='Signal type: ', title='Signal types:')
        return signal_builders[builder_name](sample_rate, force_sample_rate)

    @staticmethod
    def select_signal(env, text, signal_name=None): 
        io = None
        if signal_name is not None:
            io = IO.get(env, signal_name)

        while io is None:
            try:
                io = IO.get(env, input(text))
            except IndexError:
                print("Invalid IO name: %s, try again",signal_name)
                continue

        return io

    @staticmethod
    def menu_choice(items, prompt='Choice: ', title=None, seperator='\n', render_item=lambda x: x): #Creates menu. Can take in multiple types of menus
        if title is not None:
            print(title)
        for i, item in enumerate(items):
            print('[{}] {}'.format(i, render_item(item))) #Problems with end = separator (end indicates what goes at the end of string)

        choice = None
        while choice is None:
            user_text = input(prompt)

            try:
                user_index = int(user_text)
                choice = items[user_index]
            except ValueError: 
                print('Invalid choice: not a number')
                continue
            except IndexError:
                print('Invalid choice: out of range')
                continue

        return user_index, choice


class IO: #Input output interface
    abbrev = 'IO'

    @staticmethod
    def get(env, name):
        io_type, io_index = parse_io_name(name)

        io_types = {'i': env.inputs, 'o': env.outputs}
        if io_type not in io_types:
            return
        
        return io_types[io_type][io_index - 1]

    def __init__(self, index, mode='disabled', env=None):
        self.index = index
        self.mode = mode
        self.signal = None
        self.env = env

    def status(self):
        io_display_str = ''

        if self.mode is not None:
            io_display_str += self.mode
            if self.signal is not None:
                io_display_str += ', Signal: ' + str(self.signal)

        else:
            io_display_str += 'Not configured'

        return io_display_str

    def configure(self, args):
        _, self.mode = Interface.menu_choice(['enabled', 'disabled'], prompt='Mode: ')
        if self.mode == 'enabled':
            self.config_enable()

        elif self.mode == 'disabled':
            self.config_disable()

        print('{}{} configured: {}'.format(self.abbrev, self.index, self.status()))

    def disable(self):
        self.mode = 'disabled'
        self.config_disable()
        print('{}{} configured: {}'.format(self.abbrev, self.index, self.status()))

    def config_enable(self):
        raise NotImplementedError('Not implemented in abstract class')

    def config_disable(self):
        self.signal = None

    def __str__(self):
        return '{}{}'.format(self.abbrev, self.index)

    @property
    def enabled(self):
        return self.mode != 'disabled'


class Input(IO):
    abbrev = 'I'

    def config_enable(self):
        self.signal = Interface.build_signal(self.env.global_sample_rate)
        
#Calls IO interface
class Output(IO): 
    abbrev = 'O'

    def __init__(self, index, mode='disabled', env=None):
        super().__init__(index, mode, env)
        self.trigger = saleae.Trigger.NoTrigger

    def config_enable(self):
        _, self.trigger = Interface.menu_choice(list(saleae.Trigger), render_item=lambda x: x._name_, 
            title='Available trigger modes:', prompt='Trigger mode: ')

    def status(self):
        io_display_str = super().status()

        if self.mode == 'enabled' and self.trigger != saleae.Trigger.NoTrigger: 
            io_display_str += ', Trigger: {}'.format(self.trigger._name_)

        return io_display_str


def populate_behavior_model_signals(environment, model, *args):
    if len(model.relevant_inputs) + len(model.relevant_outputs) == 0:
        return

    if len(args) == 2:
        model.relevant_input_values = args[0]
        model.relevant_output_values = args[1]
        return

    print('============================================')
    print('Identify the signals to be used by the test:')
    model.relevant_input_values = []
    model.relevant_output_values = []

    for ri in model.relevant_inputs:
        text = ri + ': '
        while True:
            value = input(text)

            # TODO: Don't let the user set inputs as outputs and vice versa
            io = environment.get_io(value)
            if io is None:
                print("Invalid IO name: %s, try again"%value)
            else:
                model.relevant_input_values.append(io)
                break

    for ro in model.relevant_outputs:
        text = ro + ': '
        while True:
            value = input(text)

            # TODO: Don't let the user set inputs as outputs and vice versa
            io = environment.get_io(value)

            if io is None:
                print("Invalid IO name: %s, try again"%value)
            else:
                model.relevant_output_values.append(io)
                break

#Names seem explanatory. Can adjust each command individually
class Clear(Frame):
    def create_widgets(self):

        self.config_button_1 = Button(self)
        self.config_button_2 = Button(self)
        self.config_button_3 = Button(self)
        self.config_button_4 = Button(self)
        self.config_button_5 = Button(self)

        self.config_button_1['text'] = 'CONFIG0'
        self.config_button_1['fg'] = 'green'
        self.config_button_1['command'] = self.config1
        self.config_button_1.pack({'side': 'left'})

        self.config_button_2['text'] = 'CONFIG1'
        self.config_button_2['fg'] = 'green'
        self.config_button_2['command'] = self.config2
        self.config_button_2.pack({'side': 'left'})

        self.config_button_3['text'] = 'CONFIG2'
        self.config_button_3['fg'] = 'green'
        self.config_button_3['command'] = self.config3
        self.config_button_3.pack({'side': 'left'})

        self.config_button_4['text'] = 'CONFIG3'
        self.config_button_4['fg'] = 'green'
        self.config_button_4['command'] = self.config4
        self.config_button_4.pack({'side': 'left'})

        self.config_button_5['text'] = 'CONFIG4'
        self.config_button_5['fg'] = 'green'
        self.config_button_5['command'] = self.config4
        self.config_button_5.pack({'side': 'left'})

    def config1(self):
        global clearIO
        print('clear I0')
        clearIO = 'I0'
        env.clear('')
        
    def config2(self):
        global clearIO
        print('clear I1')
        clearIO = 'I1'
        env.clear('')

    def config3(self):
        global clearIO
        print('clear I2')
        clearIO = 'I2'
        env.clear('')

    def config4(self):
        global clearIO
        print('clear I3')
        clearIO = 'I3'
        env.clear('')

    def config5(self):
        global clearIO
        print('clear I4')
        clearIO = 'I4'
        env.clear('')

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.config_button = None
        self.pack()
        self.create_widgets()
        self.poll()

    def poll(self):
        """
        This method is required to allow the mainloop to receive keyboard
        interrupts when the frame does not have the focus
        """
        self.master.after(250, self.poll)


class Copy1(Frame):

    def create_widgets(self):
        copyInput = 'I0'
        toInput = 'I0'
        self.config_button_1 = Button(self)
        self.config_button_2 = Button(self)
        self.config_button_3 = Button(self)
        self.config_button_4 = Button(self)
        self.config_button_5 = Button(self)

        self.config_button_1['text'] = 'CONFIG0'
        self.config_button_1['fg'] = 'green'
        self.config_button_1['command'] = self.config1
        self.config_button_1.pack(side=LEFT)

        self.config_button_2['text'] = 'CONFIG1'
        self.config_button_2['fg'] = 'green'
        self.config_button_2['command'] = self.config2
        self.config_button_2.pack({'side': 'left'})

        self.config_button_3['text'] = 'CONFIG2'
        self.config_button_3['fg'] = 'green'
        self.config_button_3['command'] = self.config3
        self.config_button_3.pack({'side': 'left'})

        self.config_button_4['text'] = 'CONFIG3'
        self.config_button_4['fg'] = 'green'
        self.config_button_4['command'] = self.config4
        self.config_button_4.pack({'side': 'left'})

        self.config_button_5['text'] = 'CONFIG4'
        self.config_button_5['fg'] = 'green'
        self.config_button_5['command'] = self.config5
        self.config_button_5.pack({'side': 'left'})

        self.config_button_6 = Button(self)
        self.config_button_7 = Button(self)
        self.config_button_8 = Button(self)
        self.config_button_9 = Button(self)
        self.config_button_10 = Button(self)

        self.config_button_6['text'] = 'CONFIG0_TO'
        self.config_button_6['fg'] = 'green'
        self.config_button_6['command'] = self.config6
        self.config_button_6.pack()

        self.config_button_7['text'] = 'CONFIG1_TO'
        self.config_button_7['fg'] = 'green'
        self.config_button_7['command'] = self.config7
        self.config_button_7.pack()

        self.config_button_8['text'] = 'CONFIG2_TO'
        self.config_button_8['fg'] = 'green'
        self.config_button_8['command'] = self.config8
        self.config_button_8.pack()

        self.config_button_9['text'] = 'CONFIG3_TO'
        self.config_button_9['fg'] = 'green'
        self.config_button_9['command'] = self.config9
        self.config_button_9.pack()

        self.config_button_10['text'] = 'CONFIG4_TO'
        self.config_button_10['fg'] = 'green'
        self.config_button_10['command'] = self.config10
        self.config_button_10.pack()

        self.copy = Button(self)
        self.copy['text'] = 'COPY'
        self.copy['fg'] = 'red'
        self.copy['command'] = self.copyit
        self.copy.pack({'side': 'left'})

    def config1(self):
        global copyInput
        print('I0')
        copyInput = 'I0'
        
        
    def config2(self):
        global copyInput
        print('I1')
        copyInput = 'I1'
        

    def config3(self):
        global copyInput
        print('I2')
        copyInput = 'I2'
        

    def config4(self):
        global copyInput
        print('I3')
        copyInput = 'I3'
        

    def config5(self):
        global copyInput
        print('I4')
        copyInput = 'I4'

    def config6(self):
        global toInput
        print('I0')
        toInput = 'I0'
        
        
    def config7(self):
        global toInput
        print('I1')
        toInput = 'I1'
        

    def config8(self):
        global toInput
        print('I2')
        toInput = 'I2'
        

    def config9(self):
        global toInput
        print('I3')
        toInput = 'I3'
        

    def config10(self):
        global toInput
        print('I4')
        toInput = 'I4'

    def copyit(self):
        global toInput
        global copyInput
        print('copy ' + copyInput + ' to ' +toInput)
        env.copy([copyInput, toInput])
        

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.config_button = None
        self.pack()
        self.create_widgets()
        self.poll()

    def poll(self):
        """
        This method is required to allow the mainloop to receive keyboard
        interrupts when the frame does not have the focus
        """
        self.master.after(250, self.poll)

class Disable(Frame):
    def create_widgets(self):
        self.config_button_1 = Button(self)
        self.config_button_2 = Button(self)
        self.config_button_3 = Button(self)
        self.config_button_4 = Button(self)
        self.config_button_5 = Button(self)

        self.config_button_1['text'] = 'I0'
        self.config_button_1['fg'] = 'green'
        self.config_button_1['command'] = self.I0
        self.config_button_1.pack({'side': 'left'})

        self.config_button_2['text'] = 'I1'
        self.config_button_2['fg'] = 'green'
        self.config_button_2['command'] = self.I1
        self.config_button_2.pack({'side': 'left'})

        self.config_button_3['text'] = 'I2'
        self.config_button_3['fg'] = 'green'
        self.config_button_3['command'] = self.I2
        self.config_button_3.pack({'side': 'left'})

        self.config_button_4['text'] = 'I3'
        self.config_button_4['fg'] = 'green'
        self.config_button_4['command'] = self.I3
        self.config_button_4.pack({'side': 'left'})

        self.config_button_5['text'] = 'I4'
        self.config_button_5['fg'] = 'green'
        self.config_button_5['command'] = self.I4
        self.config_button_5.pack({'side': 'left'})


    def I0(self):
        print('I0')
        global disableInput
        disableInput = 'I0'
        env.disable('I0')
    def I1(self):
        print('I1')
        global disableInput
        disableInput = 'I1'
        env.disable('I0')
    def I2(self):
        print('I2')
        global disableInput
        disableInput = 'I2'
        env.disable('I0')
    def I3(self):
        print('I3')
        global disableInput
        disableInput = 'I3'
        env.disable('I0')
    def I4(self):
        print('I4')
        global disableInput
        disableInput = 'I4'
        env.disable('I0')

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.config_button = None
        self.pack()
        self.create_widgets()
        self.poll()

    def poll(self):
        """
        This method is required to allow the mainloop to receive keyboard
        interrupts when the frame does not have the focus
        """
        self.master.after(250, self.poll)

class Config(Frame):
    def create_widgets(self):

        self.config_button_1 = Button(self)
        self.config_button_2 = Button(self)
        self.config_button_3 = Button(self)
        self.config_button_4 = Button(self)

        self.config_button_1['text'] = 'CONFIG1'
        self.config_button_1['fg'] = 'green'
        self.config_button_1['command'] = self.config1
        self.config_button_1.pack({'side': 'left'})

        self.config_button_2['text'] = 'CONFIG2'
        self.config_button_2['fg'] = 'green'
        self.config_button_2['command'] = self.config2
        self.config_button_2.pack({'side': 'left'})

        self.config_button_3['text'] = 'CONFIG3'
        self.config_button_3['fg'] = 'green'
        self.config_button_3['command'] = self.config3
        self.config_button_3.pack({'side': 'left'})

        self.config_button_4['text'] = 'CONFIG4'
        self.config_button_4['fg'] = 'green'
        self.config_button_4['command'] = self.config4
        self.config_button_4.pack({'side': 'left'})

    def config1(self):
        print('I0')
        app = TextBox(master=Tk())
    def config2(self):
        print('I1')
        app = TextBox(master=Tk())
    def config3(self):
        print('I2')
        app = TextBox(master=Tk())
    def config4(self):
        print('I3')
        app = TextBox(master=Tk())

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.config_button = None
        self.pack()
        self.create_widgets()
        self.poll()

    def poll(self):
        """
        This method is required to allow the mainloop to receive keyboard
        interrupts when the frame does not have the focus
        """
        self.master.after(250, self.poll)
class TextBox(Frame):
    def create_widgets(self):
        win = Tk()
        win.title("CAERUS CNC")
        tab_control = ttk.Notebook(win)
        tab2 = ttk.Frame(tab_control)
        self.scrTxt = scrolledtext.ScrolledText(tab2,width=40,height=10)


    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master.minsize(width=400, height=240)
        self.pack()
        self.create_widgets()
        self.poll()

    def poll(self):
        """
        This method is required to allow the mainloop to receive keyboard
        interrupts when the frame does not have the focus
        """
        self.master.after(250, self.poll)


class Application(Frame):

    inputVal = 'I1'
    def create_widgets(self):
        self.status_button = Button(self)
        self.status_button['text'] = 'STATUS'
        self.status_button['fg'] = 'red'
        self.status_button['command'] = self.status
        self.status_button.pack({'side': 'left'})

        self.help_button = Button(self)
        self.help_button['text'] = 'HELP'
        self.help_button['fg'] = 'blue'
        self.help_button['command'] = self.help
        self.help_button.pack({'side': 'left'})

        self.config_button = Button(self)
        self.config_button['text'] = 'CONFIG'
        self.config_button['fg'] = 'green'
        self.config_button['command'] = self.config
        self.config_button.pack({'side': 'left'})

        self.run_button = Button(self)
        self.run_button['text'] = 'RUN'
        self.run_button['fg'] = 'red'
        self.run_button['command'] = self.run
        self.run_button.pack({'side': 'left'})

        self.copy_button = Button(self)
        self.copy_button['text'] = 'COPY'
        self.copy_button['fg'] = 'blue'
        self.copy_button['command'] = self.copy
        self.copy_button.pack({'side': 'left'})

        self.copy_button = Button(self)
        self.copy_button['text'] = 'CLEAR'
        self.copy_button['fg'] = 'green'
        self.copy_button['command'] = self.clear
        self.copy_button.pack({'side': 'left'})

        self.copy_button = Button(self)
        self.copy_button['text'] = 'DISABLE'
        self.copy_button['fg'] = 'red'
        self.copy_button['command'] = self.disable
        self.copy_button.pack({'side': 'left'})

        self.sample_button = Button(self)
        self.sample_button['text'] = 'SAMPLE'
        self.sample_button['fg'] = 'blue'
        self.sample_button['command'] = self.sample
        self.sample_button.pack({'side': 'left'})

        self.save_button = Button(self)
        self.save_button['text'] = 'SAVE'
        self.save_button['fg'] = 'red'
        self.save_button['command'] = self.save
        self.save_button.pack({'side': 'left'})

        self.load_button = Button(self)
        self.load_button['text'] = 'LOAD'
        self.load_button['fg'] = 'green'
        self.load_button['command'] = self.load
        self.load_button.pack({'side': 'left'})

    def save(self):
        print('save')
        root = tk.Tk()
        title = tk.Label(root, text="Save as...",font=("Arial",18))
        global textBox
        textBox=Text(root, height=2, width=10)
        title.pack()
        textBox.pack()
        buttonCommit=Button(root, height=1, width=10, text="Commit")
        buttonCommit['command'] = self.retrive_input
        buttonCommit.pack()
    def load(self):
        filename = filedialog.askopenfilename(filetypes = (("Template files", "*.tplate")
                                                         ,("HTML files", "*.html;*.htm")
                                                         ,("All files", "*.*") ))
        
        print(os.path.basename(filename))
        global loadFileName
        loadFileName = os.path.basename(filename)
        env.load('')

    def retrive_input(self):
        global inputValue
        inputValue=textBox.get("1.0","end-1c")
        print(inputValue)
        env.save('')
        #filename = filedialog.askopenfilename(filetypes = (("Template files", "*.tplate")
         #                                                ,("HTML files", "*.html;*.htm")
          #                                               ,("All files", "*.*") ))
    def sample(self):
        print('sample')
        #app = inputBoxSample(master = Tk())
        global sampleValue
        sampleValue = '30'
        env.sample()
    def InputBox(self):        
        dialog = tk.Toplevel(self.dialogroot)
        dialog.width = 600
        dialog.height = 100

        frame = tk.Frame(dialog,  bg='#42c2f4', bd=5)
        frame.place(relwidth=1, relheight=1)

        entry = tk.Entry(frame, font=40)
        entry.place(relwidth=0.65, rely=0.02, relheight=0.96)
        entry.focus_set()

        submit = tk.Button(frame, text='OK', font=16, command=lambda: self.DialogResult(entry.get()))
        submit.place(relx=0.7, rely=0.02, relheight=0.96, relwidth=0.3)

        root_name = self.dialogroot.winfo_pathname(self.dialogroot.winfo_id())
        dialog_name = dialog.winfo_pathname(dialog.winfo_id())

        # These two lines show a modal dialog
        self.dialogroot.tk.eval('tk::PlaceWindow {0} widget {1}'.format(dialog_name, root_name))
        self.dialogroot.mainloop()

        #This line destroys the modal dialog after the user exits/accepts it
        dialog.destroy()

        #Print and return the inputbox result
        print(self.strDialogResult)
        return self.strDialogResult

    def DialogResult(self, result):
        self.strDialogResult = result
        #This line quits from the MODAL STATE but doesn't close or destroy the modal dialog
        self.dialogroot.quit()
    def disable(self):
        print('disable')
        app = Disable(master=Tk())
    def status(self):
        print('status')
        env.status()
        print('> ')
    def help(self):
        print('help')
        env.show_help(commands.keys())
        print('> ')
    def config(self):
        print('config')
        flag = 1
        #while flag == 1: 

        #env.config('I1')
        app = Config(master=Tk())
    def run(self):
        print('run')
        env.run('2')

    def copy(self):
        print('copy')
        app = Copy1(master=Tk())
    def clear(self):
        print('clear:')
        app = Clear(master=Tk())
    

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master.minsize(width=400, height=240)
        self.quit_button = None
        self.help_button = None
        self.pack()
        self.create_widgets()
        self.poll()

    def poll(self):
        """
        This method is required to allow the mainloop to receive keyboard
        interrupts when the frame does not have the focus
        """
        self.master.after(250, self.poll)

def worker_function(quit_flag):
    counter = 0
    while not quit_flag.value:
        counter += 1
        logging.info("Work # %d" % counter)
        time.sleep(1.0)
class TestEnvironment:
    def __init__(self,master=None):
        self.profile_path = None
        self.inputs = [Input(i, env=self) for i in range(1, 5)] #Up to 5 inputs
        self.outputs = [Output(i, mode='disabled', env=self) for i in range(1, 9)] #Up to 9 inputs
        self.tests = [] #Array for added tests

        try:
            self.playback_device = FPGAPlaybackDevice() #Look in device.py file
        except IndexError:
            print('No playback device found.')
            self.playback_device = None
        self.behavior_model = None #Check behavior_model.py file
        self.global_sample_rate = 25 * 1000 * 1000
        self.recording_sample_rate = 50 * 1000 * 1000


    def status(self, args=None):
        field_values = ['\n'.join(('{} - {}'.format(io.index, io.status())) for io in self.inputs)]
        field_values += ['\n'.join(('{} - {}'.format(io.index, io.status())) for io in self.outputs)]
        field_values += ['\n'.join(str(t) for t in self.tests)] if len(self.tests) > 0 else ['None\n']

        field_values += ['Sample rate: {}'.format(display_rate(self.global_sample_rate))]
        field_values += [self.behavior_model.filename if self.behavior_model else None]

        print(STATUS_MESSAGE.format(*field_values))

    def config(self, args):
        io = Interface.select_signal(self, 'IO: ', signal_name=args[0] if len(args) > 0 else None)
        io.configure(args[1:])

    def disable(self, args):
        io = IO.get(self, disableInput)
        io.disable()
        # if len(args) == 0:
        #     io = Interface.select_signal(self, 'IO: ', signal_name=args[0] if len(args) > 0 else None)
        #     io.configure(('disabled', ))
        # else:
        #     for io_name in args:
        #         io = IO.get(self, io_name)
        #         io.configure(('disabled', ))

    def show_help(self, args=[]):
        print(HELP_MESSAGE.format(', '.join(args)))
	
    def test_add(self, args=None):
        _, T = Interface.menu_choice(device_tests.tests, title='Available tests: ', prompt='Enter a test: ')
        t = T(self)
        t.run_configure_ui()
        self.tests.append(t) #Adds test

    def test_remove(self, args=None):
        i, _ = Interface.menu_choice(self.tests, prompt='Enter a test: ') #Takes input of test. I assume it considers removing tests that aren't in.
        self.tests.pop(i) 

    def load_model(self, args):
        if args:
            filepath = args[0]
        else:
            filepath = loadFileName

        print(filepath)

        if not os.path.exists(filepath):
            print('Behavior model does not exist at path: {}'.format(filepath))
            return

        # Import model
        spec = importlib.util.spec_from_file_location("module.name", filepath)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        try:
            self.behavior_model = module.ModelClass()
        except AttributeError:
            print('Behavior model must declare a ModelClass')
            return

        self.behavior_model.filename = filepath
        populate_behavior_model_signals(self, self.behavior_model, *args[1:])

    def create_model(self, args=None):
        if args:
            filepath = args[0]
        else:
            filepath = input('File path: ')

        with open(filepath, 'w') as f:
            f.write(MODEL_TEMPLATE)

        print('Created new behavior model template at {}'.format(filepath))
        
    def run(self, args):
        iterations = int(args[0]) if len(args) > 0 else 1
        results = []

        for i in range(iterations):
            if iterations > 1:
                print('\nITERATION {}'.format(i))
            for test in self.tests:
                result = test.run(self.inputs, self.outputs)
                results.append(result)

        if iterations > 1:
            print('Test results:')
            print('Ran {} iterations'.format(iterations))
            for r in results:
                print(r)

    def plot(self, args):
        
        if len(args) == 0 or not isinstance(args,list):
            print('Provide an argument specifying what to plot: either a list (e.g., ["all"], ["inputs"]) or a signal name (e.g., I1)')
            return

        if args[0] in ('all', 'outputs', 'inputs'):
            xlim = None
            if len(args) == 3:
                xlim = [float(x) for x in args[1:]]

            defined_ios = []
            if args[0] in ('all', 'inputs'):
                defined_ios.extend([io for io in self.inputs if io.signal is not None])

            if args[0] in ('all', 'outputs'):
                defined_ios.extend([io for io in self.outputs if io.signal is not None])

            if not len(defined_ios):
                print('No IOs to plot')
                return

            if xlim is None:
                xlim = [0, max((io.signal.duration for io in defined_ios))]

            for index, io in enumerate(defined_ios):
                if io.signal is not None:
                    plt.subplot(len(defined_ios), 1, index+1)
                    io.signal.plot(xlim=xlim, show=False, show_xlabel=False)

                    plt.ylabel(str(io))

            plt.xlabel('Time (s)')
            plt.show()
            return

        if len(args) >= 1:
            io = IO.get(self, args[0])
        else:
            io = IO.get(self, input('IO: '))

        if io is not None and io.signal is not None:
            xlim = None
            if len(args) == 3:
                xlim = [float(x) for x in args[1:]]

            elif len(args) == 2 and args[1] == 'period' and isinstance(io.signal, signal.Clock):
                xlim = [0, io.signal.period]

            io.signal.plot(xlim=xlim)

        else:
            if io is None:
                print("Invalid IO, args[0]: %s"%args[0])
            else:
                print('Cannot plot IO with undefined signal')

    def sample(self, args=[]):
        if(sampleValue != -1):
        # if len(args) == 1:
            self.global_sample_rate = parse_rate(sampleValue)

        # if len(args) == 0 or self.global_sample_rate is None:
        #     self.global_sample_rate = input_rate('Set sample rate: ')

    def save(self, args):
        if args:
            filepath = args[0]

        # elif self.profile_path is not None:
        #     if prompt_yesno('Profile was loaded from {}. Overwrite? '.format(self.profile_path)):
        #         filepath = self.profile_path
        #     else:
        #         filepath = inputValue
        #         #filepath = input('File path: ')

        else:
            filepath = inputValue

        backup_pd = self.playback_device
        self.playback_device = None
        if self.behavior_model is not None:
            self.behavior_model = \
            {
                'filename': self.behavior_model.filename,
                'relevant_inputs': self.behavior_model.relevant_input_values,
                'relevant_outputs': self.behavior_model.relevant_output_values
            }

        with open(filepath, 'wb') as f: #Write Bytes
            pickle.dump(self, f) #Look at this pickle problem

        self.load([filepath], status=False)
        self.playback_device = backup_pd

    def load(self, args, status=True):
        if args:
            filepath = args[0]
        else:
            filepath = loadFileName

        with open(filepath, 'rb') as f:
            loaded_env = pickle.load(f)

            self.profile_path = filepath
            self.inputs = loaded_env.inputs
            self.outputs = loaded_env.outputs

            for io in self.inputs + self.outputs:
                io.env = self

            self.tests = loaded_env.tests
            for t in self.tests:
                t.environment = self

            if loaded_env.behavior_model is not None:
                self.load_model(
                    [loaded_env.behavior_model['filename'], 
                    loaded_env.behavior_model['relevant_inputs'], 
                    loaded_env.behavior_model['relevant_outputs']])

            self.global_sample_rate = loaded_env.global_sample_rate

        if status:
            self.status()

    def get_io(self, *args):
        return IO.get(self, *args)

    def record(self, *args):
        channels = [o.index-1 for o in self.outputs if o.enabled]
        if len(channels) == 0:
            print('No outputs configured for recording. Configure outputs to record a signal.')
            return

        rates = signal.RecordingSession.sample_rates(channels)

        rate_index, _ = Interface.menu_choice([display_rate(r, unit='S/s') for r in rates],
            title='Available sample rates:', prompt='Sample rate: ')
        dur = input_duration('Duration: ')
    
        self.recording_sample_rate = rates[rate_index]
        input('Press enter to start recording...')

        recordings = signal.Signal.record(
            sample_rate=self.recording_sample_rate, 
            max_duration=dur, 
            outputs=[o for o in self.outputs if o.enabled])

        index = 0
        for i, o in zip(self.inputs, self.outputs):
            if o.enabled:
                o.signal = recordings[index]
                o.mode = 'enabled'
               
                index += 1

    def get_user_signal(self, sample_rate):
        j, _ = Interface.menu_choice(['New...', 'Existing IO signal'], title='Select signal:')

        if j == 0:
            other = Interface.build_signal(sample_rate, force_sample_rate=True)
        elif j == 1:
            other = Interface.select_signal(self)

            # TODO: Resample selected signal to requested sample rate if forced

        return other

    def edit(self, args):
        io = Interface.select_signal(self, 'IO: ', signal_name=args[0] if len(args) > 0 else None)

        if io.signal is None:
            print('IO has no singal attached; use "config" to create a signal.')
            return

        i, _ = Interface.menu_choice(['Append', 'Prepend', 'Truncate', 'Set sample rate', 'Scale'])

        if i == 0:
            io.signal = io.signal.append(self.get_user_signal(io.signal.sample_rate)) #appends new signal

        elif i == 1:
            io.signal = self.get_user_signal(io.signal.sample_rate).append(io.signal) #prepend new signal

        elif i == 2: #Truncate by setting new start and end times
            t0 = input_duration('Start time: ')
            t1 = input_duration('End time: ')
            io.signal = io.signal.truncate(t0, t1)

        elif i == 3: #New sample rate
            rate = input_rate('Sample rate: ')
            io.signal.sample_rate = rate

    def copy(self, args):
        # copyInput is not '' and toInput is not '':
        print(copyInput)
        print(toInput)
        io1 = IO.get(env, copyInput)
        io2 = IO.get(env, toInput)
        #else:
        #    io1 = Interface.select_signal(self, 'From IO: ', signal_name=args[0] if len(args) > 0 else None)
        #    io2 = Interface.select_signal(self, 'To IO: ', signal_name=args[1] if len(args) > 1 else None)

        io2.mode = io1.mode

        if io1.signal is None:
            io2.mode = 'disabled'
            io2.signal = None
        else:
            io2.mode = 'enabled'
            io2.signal = io1.signal.clone()

    def clear(self, args):
        #io = Interface.select_signal(self, 'IO: ', signal_name=args[0] if len(args) > 0 else None)
        global clearIO
        io = IO.get(env,clearIO)
        print(clearIO)
        io.signal = None

#Parsing input
def parse_io_name(name):
    if len(name) == 0:
        return None

    io_type = name[0].lower()

    if io_type not in ('i', 'o'):
        print(IO_NAME_MESSAGE)
        return None, None

    try:
        io_index = int(name[1:])
    except ValueError:
        print(IO_NAME_MESSAGE)
        return None, None

    return io_type, io_index

def input_int(*args):
    while 1:
        val = input(*args) #For integers
        try:
            return int(val)
        except ValueError:
            print('Not an integer, try again')


def main():
    sampleValue = -1
    #env = TestEnvironment()
    
    commands = {
        'help': lambda a: env.show_help(commands.keys()), #lambda procedure:expression
        'status': env.status,
        'config': env.config,
        'disable': env.disable,
        'test_add': env.test_add,
        'test_remove': env.test_remove,
        'model_create': env.create_model,
        'model_load': env.load_model,
        'run': env.run,
        'plot': env.plot,
        'sample': env.sample,
        'save': env.save,
        'load': env.load,
        'record': env.record,
        'copy': env.copy,
        'edit': env.edit,
        'clear': env.clear
    }

    if len(sys.argv) >= 2 and os.path.exists(sys.argv[1]): #Checking for all arguments entered and for valid arguments
        env.load([sys.argv[1]])
        print('{} loaded.'.format(sys.argv[1]))

    else:
        env.show_help(commands.keys())
        print('')
        env.status()
    root = Tk()
    app = Application(master=root)
    print('')
    # p=Popen(["./sample.sh"],stdin=PIPE)
    # cmd, *args = raw_input()
    # p.stdin.write(line+'\n')
    
    try:
        while 1:
            cmd=' '
            args = ""
            cmd, *args = input('> ').split(' ') #Performs commands with arguments 

            if cmd == ' ':
                
                pass

            elif cmd in commands:
                try:
                    commands[cmd](args) #Performs given command
                except KeyboardInterrupt:
                    pass

            elif cmd == 'exit':
                print(EXIT_MESSAGE)
                break
            else:
                print('No such command: "{}"'.format(cmd))
                env.show_help(commands.keys())

            #print('')
            


    except (KeyboardInterrupt, EOFError):
        print('\n' + EXIT_MESSAGE)


    except exceptions.PlaybackDeviceException as e:
        print('Playback peripheral: ' + str(e))
        print('Aborting.')

    except:
        from datetime import datetime
        env.save(('_crash_' + (datetime.today().isoformat().replace(':', '.')) + '.profile', ))
        import traceback
        traceback.print_exc()
        raise

env = TestEnvironment()

"""
format = '%(levelname)s: %(filename)s: %(lineno)d: %(message)s'
logging.basicConfig(level=logging.DEBUG, format=format)
root = Tk()
app = Application(master=root)
quit_flag = multiprocessing.Value('i', int(False))
worker_thread = threading.Thread(target=worker_function, args=(quit_flag,))
worker_thread.start()
logging.info("quit_flag.value = %s" % bool(quit_flag.value))
try:
    app.mainloop()
except KeyboardInterrupt:
    logging.info("Keyboard interrupt")
quit_flag.value = True
logging.info("quit_flag.value = %s" % bool(quit_flag.value))
worker_thread.join()
"""
commands = {
        'help': lambda a: env.show_help(commands.keys()), #lambda procedure:expression
        'status': env.status,
        'config': env.config,
        'disable': env.disable,
        'test_add': env.test_add,
        'test_remove': env.test_remove,
        'model_create': env.create_model,
        'model_load': env.load_model,
        'run': env.run,
        'plot': env.plot,
        'sample': env.sample,
        'save': env.save,
        'load': env.load,
        'record': env.record,
        'copy': env.copy,
        'edit': env.edit,
        'clear': env.clear
        }
