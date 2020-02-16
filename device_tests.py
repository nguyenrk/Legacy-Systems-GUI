import string
import time
import numpy as np

from . import signal
from .exceptions import PlaybackDeviceException
from .utils import parse_rate, parse_duration, parse_percent, binary_search


class TestParameter:
    def __init__(self, display_name, units, obj_cast=float):
        self.display_name = display_name
        self.units = units
        self.obj_cast = obj_cast


class TestIOMapping:
    def __init__(self, display_name, optional=False):
        self.display_name = display_name
        self.optional = optional


class DeviceTestMeta(type):
    def __str__(self):
        return self.test_name


class DeviceTest(metaclass=DeviceTestMeta):
    test_name = None
    parameters = []
    relevant_inputs = []
    relevant_outputs = []

    def __init__(self, environment):
        self.environment = environment
        self.parameter_values = []
        self.relevant_input_values = []
        self.relevant_output_values = []

    @property
    def behavior_model(self):
        return self.environment.behavior_model

    @property
    def device(self):
        return self.environment.playback_device

    def __str__(self):
        text = '{}\n'.format(self.test_name)

        if len(self.parameters):
            text += '    ' + (', '.join('{}: {}'.format(p.display_name, str(val)) \
                    for p, val in zip(self.parameters, self.parameter_values)))
            text += '\n'

        if len(self.relevant_inputs):
            text += '    ' + (', '.join('{}: {}'.format(ri.display_name, str(val)) \
                    for ri, val in zip(self.relevant_inputs, self.relevant_input_values)))
            text += '\n'

        if len(self.relevant_outputs):
            text += '    ' + (', '.join('{}: {}'.format(ro.display_name, str(val)) \
                    for ro, val in zip(self.relevant_outputs, self.relevant_output_values)))
            text += '\n'
        
        return text

    def run_configure_ui(self):
        self.parameter_values = []
        self.relevant_input_values = []
        self.relevant_output_values = []

        if len(self.parameters):
            print('============================================')
            print('Test parameters:')
            for p in self.parameters:
                text = p.display_name + ' (' + ', '.join(u for u in p.units) + '): '

                while True:
                    try:
                        value = p.obj_cast(input(text))
                        break
                    except ValueError:
                        print('Invalid value, try again')

                self.parameter_values.append(value)
            print()

        if len(self.relevant_inputs) + len(self.relevant_outputs):
            print('============================================')
            print('Identify the signals to be used by the test: (inputs first)')
            for ri in self.relevant_inputs:
                text = ri.display_name + ': '
                while True:
                    value = input(text)

                    # TODO: Don't let the user set inputs as outputs and vice versa
                    io = self.environment.get_io(value)

                    # TODO: We should probably be able to tell the difference between a user wanted to set it to None,
                    # and when a user just gives a invalid input name
                    if ri.optional and io is None:
                        self.relevant_input_values.append(None)
                    elif io is None:
                        print('Invalid IO name, try again')
                    else:
                        self.relevant_input_values.append(io)
                        break

            for ro in self.relevant_outputs:
                text = ro.display_name + ': '
                while True:
                    value = input(text)

                    # TODO: Don't let the user set inputs as outputs and vice versa
                    io = self.environment.get_io(value)

                    # TODO: We should probably be able to tell the difference between a user wanted to set it to None,
                    # and when a user just gives a invalid input name
                    if ro.optional and io is None:
                        self.relevant_output_values.append(None)
                    elif io is None:
                        print('Invalid IO name, try again')
                    else:
                        self.relevant_output_values.append(io)
                        break
            print()

    def send_inputs(self, inputs, outputs):
        print('    Send inputs...')
        signals = [i.signal if i.enabled else None for i in inputs]

        if self.device is None:
            raise PlaybackDeviceException('No playback peripheral connected.')

        self.device.load_signals(signals)

        # TODO: remove duration calcs; device should auto-stop when playback done
        #if all(s.repeat for s in signals):
        #    duration = self.device.MAX_LENGTH / max(s.sample_rate for s in signals)
        #else:
        #    duration = min((s.sample_count / s.sample_rate for s in signals if not s.repeat))

        #print('Recording for duration:', duration)

        # TODO: USE ONLY SUPPORTED SAMPLE RATES
        session = signal.Signal.start_recording(
                sample_rate=self.environment.recording_sample_rate, 
                outputs=[o for o in outputs if o.enabled])

        # TODO: Use device stop capabilities
        # Wait for duration of shortest signal
        print('Playing...')
        self.device.play() 
            
        #time.sleep(duration)
        #self.device.stop()

        signals = session.finish_recording()

        # Save output signals
        i = 0
        for o in outputs:
            if o.enabled:
                o.signal = signals[i]
                i += 1

    def run(self, inputs, outputs):
        pass


class BasicPlayback(DeviceTest):
    """Plays back the loaded signals; does nothing else."""
    test_name = 'Basic Playback'

    def run(self, inputs, outputs):
        self.send_inputs(inputs, outputs)

        print('Analyzing...')
        if self.behavior_model is not None:
            if self.behavior_model.validate(inputs, outputs):
                print('Device behavior validated!')
                print(outputs)

            else:
                print('Device behavior did not validate.')


class ResponseTimeTest(DeviceTest):
    test_name = 'Response Time'

    relevant_inputs = [
        TestIOMapping('Input trigger')
        
    ]
    relvant_outputs = [
        TestIOMapping('Output trigger')
    ]

    def run(self, inputs, outputs):
        i = self.relevant_input_values[0]
        o = self.relevant_output_values[0]

        self.send_inputs(inputs, outputs)

        input_groups = i.signal.find_groups(False, 8)
        output_groups = o.signal.find_groups(False, 8)

        response_start_time = output_groups[0][0]

        for input_group in input_groups[::-1]:
            if input_group[1] < response_start_time:
                break

        return response_start_time - input_group[1]


# The clock rate test takes a range of frequencies, and uses binary search to
#   determine how much flexibility there is in the different clock speeds that
#   can be used. The test assumes that frequencies will behave consistently when
#   in the valid clock range, and there is some threshold that, once crossed,
#   will result in behavior that does not reflect that outlined in the behavioral
#   model. The configurable inputs are the range of of input frequencies and the
#   number of iterations the test will run.

class ClockRateTest(DeviceTest):
    test_name = 'Clock Rate'

    parameters = [
        TestParameter('Minimum Frequency', ('Hz', 'kHz', 'MHz'), parse_rate),
        TestParameter('Maximum Frequency', ('Hz', 'kHz', 'MHz'), parse_rate),
        # TestParameter('Precision', ('Hz', '%'), float),
    ]

    relevant_inputs = [
        TestIOMapping('Reset', optional=True),
        TestIOMapping('Clock')
    ]

    def run(self, inputs, outputs, numRuns=10):
        freq_min = self.parameter_values[0]
        freq_max = self.parameter_values[1]

        #Input integrity validation
        if freq_min < 0:
            print("Invalid minimum frequency value.")
            return
        if freq_max < 0:
            print("Invalid maximum frequency value.")
            return
        if freq_min > freq_max:
            print("Maximum frequency value less than minimum frequency value.")
            return
        if numRuns <= 0:
            print("Number of iterations too low.")
            return

        # Initialization
        reset = self.relevant_input_values[0]
        clock = self.relevant_input_values[1]

        if reset.signal is None:
            reset.signal = signal.Pulse(signal.LOW, 0.100, 0.100, 1.0)

        freq = (freq_min + freq_max)/2.0
        minBounds = [-1, -1]
        maxBounds = [-1, -1]
        print("Checking the overall frequency range of [{},{}]".format(freq_min,freq_max))

        #Consider preliminary check for freq_min/freq_max

        #Check if middle is valid, otherwise find an valid start freq
        clock.signal = signal.Clock(freq, duty_cycle=clock.signal.duty_cycle)
        self.send_inputs(inputs, outputs)
        if self.behavior_model.validate(inputs, outputs):
            minBounds = [freq_min, freq]
            maxBounds = [freq, freq_max]
            print("Found starting frequency value of {}".format(freq))
        else:
            found = False
            iters = 2
            
            # With no information on the signal, we must use depth first search
            # in attempts to find a valid start location.
            for i in range(numRuns):
                if found:
                    break
                else:
                    scale = (freq_max - freq_min)/(iters + 1)
                    for j in range(1,iters+1,1):
                        freq = freq_min + j*scale
                        clock.signal = signal.Clock(freq, duty_cycle=clock.signal.duty_cycle)
                        self.send_inputs(inputs, outputs)
                        if self.behavior_model.validate(inputs, outputs):
                            minBounds = [freq_min, freq]
                            maxBounds = [freq, freq_max]
                            print("Found starting frequency value of {}".format(freq))
                            
                            found = True    #Need to break 2 loops
                            break
                iters *= 2
        
        if minBounds[0] == -1:
            print("Unable to find a valid start frequency. Possible reasons:\n1.) Given region too large or number of iterations too low.\n2.) No valid freqencies inside given region.")
        
        # Expand acceptable clock frequency range with binary search
        for i in range(numRuns):
            # Expanding towards rising clock edge
            freq1 = (minBounds[0] + minBounds[1])/2.0
            clock.signal = signal.Clock(freq1, duty_cycle=clock.signal.duty_cycle)
            self.send_inputs(inputs, outputs)
            if self.behavior_model.validate(inputs, outputs):
                minBounds[1] = freq1
            else:
                minBounds[0] = freq1
            
            # Expanding toward falling clock edge
            freq2 = (maxBounds[0] + maxBounds[1])/2.0
            print("Checking MIN value of {} Checking MAX value of {}".format(freq1,freq2))
            clock.signal = signal.Clock(freq2, duty_cycle=clock.signal.duty_cycle)
            self.send_inputs(inputs, outputs)
            if self.behavior_model.validate(inputs, outputs):
                maxBounds[0] = freq2
            else:
                maxBounds[1] = freq2
        
        print("After {} iterations, the acceptable consistent clock range is [{}, {}]. The entire acceptable range is encompassed within the range of [{} {}]".format(numRuns, minBounds[1], maxBounds[0],minBounds[0],maxBounds[1]))
        if minBounds[0] == freq_min:
            print("NOTE: the minimum clock speed may be less than {}".format(freq_min))
        if maxBounds[1] == freq_max:
            print("NOTE: the maximum clock speed may be greater than {}".format(freq_max))

        # TODO: return result dict (see ButtonPulseWidthTest... used in TestEnviornment.run for multi-iteration runs)

# The duty cycle test takes a range of duty cycles, and uses binary search to
#   determine how much flexibility there is in the different duty cycles that
#   can be used. The test assumes that duty cycles will behave consistently when
#   in the valid range, and there is some threshold that, once crossed,
#   will result in behavior that does not reflect that outlined in the behavioral
#   model. The configurable inputs are the range of of input duty cycles and the
#   number of iterations the test will run.

class DutyCycleTest(DeviceTest):
    test_name = 'Duty Cycle Test'

    parameters = [
        TestParameter('Minimum Duty Cycle', ('%'), parse_percent),
        TestParameter('Maximum Duty Cycle', ('%'), parse_percent),
    ]

    relevant_inputs = [
        TestIOMapping('Reset', optional=True),
        TestIOMapping('Clock')
    ]

    def run(self, inputs, outputs, numRuns=10):
        duty_min = self.parameter_values[0]
        duty_max = self.parameter_values[1]

        #Input integrity validation    
        if duty_min < 0:
            print("Invalid minimum duty cycle value.")
            return
        if duty_max < 0:
            print("Invalid maximum duty cycle value.")
            return
        if duty_max > 100:
            print("Duty cycle value too high. Please use a duty cycle of 0.0 - 100.0")
            return
        if duty_min > duty_max:
            print("Maximum duty cycle value less than minimum duty cycle value.")
            return
        if numRuns <= 0:
            print("Number of iterations too low.")
            return

        # Initialization
        reset = self.relevant_input_values[0]
        clock = self.relevant_input_values[1]
        if reset.signal is None:
            reset.signal = signal.Pulse(signal.LOW, 0.100, 0.100, 1.0)

        duty = (duty_min + duty_max)/2.0
        minBounds = [-1, -1]
        maxBounds = [-1, -1]

        #Check if middle is valid, otherwise find an valid start duty cycle
        clock.signal = signal.Clock(clock.signal.clock_rate, duty_cycle=duty, sample_rate=clock.signal.sample_rate)
        self.send_inputs(inputs, outputs)
        if self.behavior_model.validate(inputs, outputs):
            minBounds = [duty_min, duty]
            maxBounds = [duty, duty_max]
            print("Found starting duty cycle value of {}".format(duty))
        else:
            found = False
            iters = 2
            for i in range(numRuns):
                if found:
                    break
                else:
                    scale = (duty_max - duty_min)/(iters + 1)
                    for j in range(1,iters+1,1):
                        duty = duty_min + j*scale
                        clock.signal = signal.Clock(clock.signal.clock_rate, duty_cycle=duty, sample_rate=clock.signal.sample_rate)
                        self.send_inputs(inputs, outputs)
                        if self.behavior_model.validate(inputs, outputs):
                            minBounds = [duty_min, duty]
                            maxBounds = [duty, duty_max]
                            print("Found starting duty cycle value of {}".format(duty))
                            found = True
                            break
                iters *= 2
        
        if minBounds[0] == -1:
            print("Unable to find a valid start duty cycle. Possible reasons:\n1.) Given region too large or number of iterations too low.\n2.) No valid duty cycles inside given region.")
        
        # Expand acceptable duty cycle range with binary search
        for i in range(numRuns):
            duty1 = (minBounds[0] + minBounds[1])/2.0
            clock.signal = signal.Clock(clock.signal.clock_rate, duty_cycle=duty1, sample_rate=clock.signal.sample_rate)
            self.send_inputs(inputs, outputs)
            if self.behavior_model.validate(inputs, outputs):
                minBounds[1] = duty1
            else:
                minBounds[0] = duty1
            
            duty2 = (maxBounds[0] + maxBounds[1])/2.0
            print("Checking MIN value of {} Checking MAX value of {}".format(duty1,duty2))
            clock.signal = signal.Clock(clock.signal.clock_rate, duty_cycle=duty2, sample_rate=clock.signal.sample_rate)
            self.send_inputs(inputs, outputs)
            if self.behavior_model.validate(inputs, outputs):
                maxBounds[0] = duty2
            else:
                maxBounds[1] = duty2
        print("After {} iterations, the acceptable consistent duty cycle range is [{}, {}]. The entire acceptable range is encompassed within the range of [{} {}]".format(numRuns, minBounds[1], maxBounds[0],minBounds[0],maxBounds[1]))
        if minBounds[0] == duty_min and duty_min != 0:
            print("NOTE: the minimum duty cycle may be less than {}".format(duty_min))
        if maxBounds[1] == duty_max and duty_max != 100:
            print("NOTE: the maximum duty cycle may be greater than {}".format(duty_max))

        # TODO: return result dict (see ButtonPulseWidthTest... used in TestEnviornment.run for multi-iteration runs)


class MaxDriftTest(DeviceTest):
    test_name = 'Max Drift Test'
    
    relevant_inputs = [
        TestIOMapping('Signal 1'),
        TestIOMapping('Signal 2'),
        TestIOMapping('Reset', optional=True)
    ]

    parameters = [
        TestParameter('Max delay on Signal 1', ('s', 'ms', 'us'), parse_duration),
        TestParameter('Max delay on Signal 2', ('s', 'ms', 'us'), parse_duration),
    ]

    def run(self, inputs, outputs, numRuns=10):
        pos_delay_max = self.parameter_values[0]
        neg_delay_max = self.parameter_values[1]

        if(pos_delay_max < 0):
            print("Invalid positive delay, please enter a positive number in seconds")
        if(neg_delay_max < 0):
            print("Invalid negative delay, please enter a positive number in seconds")


        # Parameter Validation Check
        og_signals = [self.relevant_input_values[0].signal.clone(), self.relevant_input_values[1].signal.clone()]

        self.send_inputs(inputs, outputs)
        if not self.behavior_model.validate(inputs, outputs):
            print("Unable to validate signals before attempting to drift")
            return

        
        pos_bounds = [0, pos_delay_max]
        neg_bounds = [0, neg_delay_max]

        for i in range(numRuns):
            
            delay_pos = (pos_bounds[0] + pos_bounds[1])/2.0

            s_pos = og_signals[1].insert_delay(delay_pos)
            self.relevant_input_values[0].signal = og_signals[0]
            self.relevant_input_values[1].signal = s_pos

            self.send_inputs(inputs,outputs)
            if self.behavior_model.validate(inputs, outputs):
                pos_bounds[0] = delay_pos
            else:
                pos_bounds[1] = delay_pos

            delay_neg = (neg_bounds[0] + neg_bounds[1])/2.0
            print("Checking POS value of {} Checking NEG value of {}".format(delay_pos,delay_neg))
            s_neg = og_signals[0].insert_delay(delay_neg)
            self.relevant_input_values[0].signal = s_neg
            self.relevant_input_values[1].signal = og_signals[1]

            self.send_inputs(inputs,outputs)
            if self.behavior_model.validate(inputs, outputs):
                neg_bounds[0] = delay_neg
            else:
                neg_bounds[1] = delay_neg

        if pos_bounds[0] == 0 and neg_bounds[0] == 0:
            print("Unable to find any drift values with valid signals. Possible reasons:\n1.) Given region too large or number of iterations too low.\n2.) System too precise to allow drift.")
        else: 
            print("After {} iterations, the range of acceptable delay allowed on signal 2 is [-{}, {}]. The total acceptable range is encompassed in [-{}, {}]".format(numRuns, neg_bounds[0], pos_bounds[0],neg_bounds[1],pos_bounds[1]))
            if neg_bounds[0] == neg_delay_max:
                print("NOTE: the negative drift may be greater than -{}".format(delay_neg))
            if pos_bounds[0] == pos_delay_max:
                print("NOTE: the positive drift may be greater than {}".format(delay_pos))

        # TODO: return result dict (see ButtonPulseWidthTest... used in TestEnviornment.run for multi-iteration runs)

class MaxGlitchDuration(DeviceTest):
    test_name = 'Max Glitch Duration'

    parameters = [
        TestParameter('Duration', ('s', 'cycles'), float)
    ]

    relevant_inputs = [
        TestIOMapping('Reset', optional=True),
        TestIOMapping('Clock')
    ]

    def run(self, inputs, outputs, glitch_loc=50,numRuns=10, scaled=False):
        clock = self.relevant_input_values[0]
        reset = self.relevant_input_values[1]

        if reset.signal is None:
            reset.signal = signal.Pulse(signal.LOW, 0.100, 0.100, 1.0)

        if glitch_loc > 100 or glitch_loc < 0:
            print("Invalid location, input a percentage value in the range of 0 to 100")
            return

        if numRuns <= 0:
            print("Number of iterations too low.")
            return

        if glitch_loc > 50:
            max_spread = 100 - glitch_loc
        else:
            max_spread = glitch_loc

        clock.signal = signal.Clock(freq, duty_cycle=clock.signal.duty_cycle)

        self.send_inputs(inputs, outputs)
        if not self.behavior_model.validate(inputs, outputs):
            print("Initial signal not valid.\n")
            return

        scale = 0.5*clock.signal.period*0.01
        lowBound = [glitch_loc, 0] # PERIOD CALC===================
        highBound = [glitch_loc, 100] # PERIOD CALC===================
        spreadBound = [0, max_spread] # PERIOD CALC===================

        for i in range(numRuns):

            # Step 1: Expand towards rising edge

            # Rebuild the clock without glitches
            clock.signal = signal.Clock(clock_rate=clock.signal.clock_rate,duration=clock.signal.duration)

            glitch_size_1 = (lowBound[0] + lowBound[1])/2.0 
            #TODO: GENERATE SIGNAL THAT IS LOW FROM (glitchloc - glitch_size_1, glitch_loc) MAYBE SCALE DOWN BY 0.5 TO ACCOUNT FOR PERIOD
            self.send_inputs(inputs, outputs)
            if self.behavior_model.validate(inputs, outputs):
                lowBound[0] = glitch_size_1
            else:
                lowBound[1] = glitch_size_1

    
            # Step 2: Expand towards negative edge

            # Rebuild the clock without glitches
            clock.signal = signal.Clock(clock_rate=clock.signal.clock_rate,duration=clock.signal.duration)
            glitch_size_2 = (highBound[0] + highBound[1])/2.0
            #TODO: GENERATE SIGNAL THAT IS LOW FROM (glitchloc, glitch_loc + glitch_size_2)
            self.send_inputs(inputs, outputs)
            if self.behavior_model.validate(inputs, outputs):
                highBound[0] = glitch_size_2
            else:
                highBound[1] = glitch_size_2

            # Step 3: Expand outward in both directions

            # Rebuild the clock without glitches
            clock.signal = signal.Clock(clock_rate=clock.signal.clock_rate,duration=clock.signal.duration)
            glitch_size_3 = (spreadBound[0] + spreadBound[1])/2.0
            print("Checking LOW value of -{} Checking HIGH value of {} Checking SPREAD value of +/-{}".format(glitch_size_1,glitch_size_2,glitch_size_3))
            #TODO: GENERATE SIGNAL THAT IS LOW FROM (glitchloc - glitch_size_3, glitch_loc + glitch_size_3)
            self.send_inputs(inputs, outputs)
            if self.behavior_model.validate(inputs, outputs):
                highBound[0] = glitch_size_3
            else:
                highBound[1] = glitch_size_3

        print("After {} iterations,".format(numRuns))
        print("A glitch towards the rising edge of the signal extended [{},{}] for a maximum of {} [found to not exceed {}]".format(lowBound[0],glitch_loc,(glitch_loc-lowBound[0]),lowBound[1]))
        print("A glitch towards the falling edge of the signal extended [{},{}] for a maximum of {} [found to not exceed {}]".format(glitch_loc,highBound[0],(highBound[0]-glitch_loc),highBound[1]))
        print("A glitch spreading equally in both directions extended [{},{}] for a maximum of {} [found to not exceed {}]".format((glitch_loc-spreadBound[0]),(glitch_loc+spreadBound[0]),(2*spreadBound[0]),(2*spreadBound[1])))
        if scaled == True:
            print("\nScaled based on a period of {} seconds.\nAfter {} iterations,".format(clock.signal.period,numRuns))
            print("A glitch towards the rising edge of the signal extended [{},{}] for a maximum of {} seconds [found to not exceed {} seconds]".format(lowBound[0]*scale,glitch_loc*scale,(glitch_loc-lowBound[0])*scale,lowBound[1]*scale))
            print("A glitch towards the falling edge of the signal extended [{},{}] for a maximum of {} seconds [found to not exceed {} seconds]".format(glitch_loc*scale,highBound[0]*scale,(highBound[0]-glitch_loc)*scale,highBound[1]*scale))
            print("A glitch spreading equally in both directions extended [{},{}] for a maximum of {} seconds [found to not exceed {} seconds]".format((glitch_loc-spreadBound[0])*scale,(glitch_loc+spreadBound[0])*scale,(2*spreadBound[0])*scale,(2*spreadBound[1])*scale))

        # TODO: return result dict (see ButtonPulseWidthTest... used in TestEnviornment.run for multi-iteration runs)

class MaxNumGlitchTest(DeviceTest):
    test_name = 'Max Num Glitch Test'

    parameters = [
        TestParameter('Duration', ('s', 'cycles'), float)
    ]

    def run(self, inputs, outputs, limit=10):
        clock = self.relevant_input_values[0]
        reset = self.relevant_input_values[1]

        if reset.signal is None:
            reset.signal = signal.Pulse(signal.LOW, 0.100, 0.100, 1.0)

        glitch_locations = [0.125, 0.25, 0.75]
        for loc in glitch_locations:
            # TODO: Binary search of glitch sizes would be way better

            for glitch_size in (0.25/glitch_size_count * i for i in range(1, glitch_size_count)):
                # Rebuild the clock without glitches
                clock.signal = signal.Clock(
                        clock_rate=clock.signal.clock_rate, 
                        duration=clock.signal.duration)

                print('Trying glitch size {} at loc {}'.format(glitch_size, loc))
                for period_index in range(0, round(clock.signal.duration / clock.signal.period)):
                    clock.signal.glitch((loc + period_index) * clock.signal.period, glitch_size)

                clock.signal.plot()

                error_count = 0
                for i in range(3):
                    self.send_inputs(inputs, outputs)

                    if not self.behavior_model.validate(inputs, outputs):
                        error_count += 1

                if error_count == 3:
                    print('Not work')
                    break

        # TODO: return result dict (see ButtonPulseWidthTest... used in TestEnviornment.run for multi-iteration runs)
class CanBusDeviceTest(DeviceTest):
    test_name = 'Can Bus Device'


    def run(self, inputs, outputs):
        print(inputs[0])
        time.sleep(2)
        self.send_inputs(inputs, outputs)
        print("Done")

        # self.send_inputs(inputs, outputs)

        # print('Analyzing...')
        # if self.behavior_model is not None:
        #     if self.behavior_model.validate(inputs, outputs):
        #         print('Device behavior validated!')
        #         print(outputs)

        #     else:
        #         print('Device behavior did not validate.')


class ButtonPulseWidthTest(DeviceTest):
    test_name = 'Button Pulse Width'

    parameters = [
        TestParameter('Minimum Pulse', ('s', 'ms', 'us'), parse_duration),
        TestParameter('Maximum Pulse', ('s', 'ms', 'us'), parse_duration),
    ]

    relevant_inputs = [
        TestIOMapping('Reset'),
        TestIOMapping('Button')
    ]

    def run(self, inputs, outputs, precision=10e-6, duration=0.30, setup_time=0.06):  # old prevision = 40e-9

        pulses = []
    
        #number of iterations
        iters = 70
        #iters = 30
        for i in range(iters):
            pulse_min = self.parameter_values[0]
            pulse_max = self.parameter_values[1]

            #precision = 40e-9

            # Initialization
            reset = self.relevant_input_values[0]
            button = self.relevant_input_values[1]
            val = (pulse_min + pulse_max) / 2
            search_size = val
            found_end = False

            pulse_bad_max = pulse_min  # Biggest value that didn't work
            pulse_good_min = pulse_max # Smallest vlaue that worked
            err_bad = 0
            err_good = 0

            # Find min pulse width
            while 1:
                search_size = search_size / 2
                # We can of course only send integer samples, and we must also send an integer as a sample rate. This means there will be some error
                # in the duration of the pulse we produce, due to limited precision in the sample rate on one side,
                # and in limited precision of sampling on the other. 
                # So we must ensure our error is less than our desired precision
                sr = 1/val
                while 1:
                    pulse_sample_count = round(round(sr) * val)
                    real_val = (pulse_sample_count / round(sr))
                    err = abs(val - real_val)
                    total_sample_count = round(round(sr) * duration)

                    if err < precision and sr > 1000:
                        break
                    
                    if total_sample_count > 2**15:
                        # Can't get any better precision
                        break

                    sr = sr * 2

                print("In iteration %d of %d"%(i, iters)) 
                print('Requested pulse duration: {:02f} s'.format(val))
                print('Real pulse duration: {:02f} s'.format(real_val))
                print('Sample count total:', total_sample_count)
                print('Sample count used for pulse:', pulse_sample_count)
                print('Sample rate:', round(sr))

                button.signal = signal.Pulse(signal.LOW, setup_time, val, duration - setup_time - val, sample_rate=sr)
                self.send_inputs(inputs, outputs)

                if self.behavior_model.validate(inputs, outputs):
                    if val < pulse_good_min:
                        pulse_good_min = real_val

                    val -= search_size
                else:
                    if val > pulse_bad_max:
                        pulse_bad_max = real_val

                    found_end = True
                    val += search_size

                #TODO: remove
                #self.environment.plot(['all'])

                if search_size * 2 < precision:
                    break

                if found_end:
                    pulse_min = val

            print('Minimum pulse duration in range:', pulse_bad_max, pulse_good_min)
            pulses.append(pulse_good_min)

        print(pulses)
        return {
            'invalid_pulse_max': pulse_bad_max,
            'valid_pulse_min': pulse_good_min
        }

class SingleInstructionGlitch(DeviceTest):
    test_name = 'Single Instruction Glitch'

    parameters = [
        #TestParameter('Duration', ('s', 'cycles'), float)
    ]

    relevant_inputs = [
        TestIOMapping('Reset'),
        TestIOMapping('Clock')
    ]

    def run(self, inputs, outputs):
        reset = self.relevant_input_values[0]
        clock = self.relevant_input_values[1]

        clock_rate = 1000000

        reset.signal = signal.Signal(initial_value=signal.LOW, sample_rate=2*clock_rate, duration=3/clock_rate)
        reset.signal = reset.signal.append(signal.Signal(initial_value=signal.HIGH, sample_rate=2*clock_rate, duration=13/clock_rate))

        clock.signal = signal.Clock(clock_rate)\
            .unroll(30)\
            .glitch(24.04 / clock_rate, 0.02 / clock_rate, resample=50)\
            .glitch(25.04 / clock_rate, 0.02 / clock_rate)\
            .glitch(26.04 / clock_rate, 0.02 / clock_rate)\
            .glitch(27.04 / clock_rate, 0.02 / clock_rate)

        #self.environment.plot('inputs')

        self.send_inputs(inputs, outputs)

        t1 = next(outputs[0].signal.edges('rising'))        # End of reset
        t2 = list(outputs[2].signal.edges('rising'))[-1]    # BSF complete
        

        t1b = t1 + 23 /clock_rate
        t3 = t2 + (t2 - t1b) / 4

        self.environment.plot(['outputs', str(t1b), str(t3)])
        print('Clock period count:', (t2 - t1) * clock_rate)


class SerialFuzzer(DeviceTest):
    test_name = 'Serial Fuzzer'

    parameters = [
    ]

    relevant_inputs = [
        TestIOMapping('Serial')
    ]
    relevant_outputs = [
        TestIOMapping('Reset'),
        TestIOMapping('RecordedSerial'),
        TestIOMapping('Valid')
    ]

    def run(self, inputs, outputs): 
        serialout = self.relevant_input_values[0]
        reset = self.relevant_input_values[0]
        recorded_serial = self.relevant_output_values[1]
        valid = self.relevant_output_values[2]

        password = ''
        while 1:
            durations = []

            for c in string.digits:
                data = password + c + '\n'
                serialout.signal = signal.Signal(initial_value=signal.HIGH, duration=0.05, sample_rate=9600).append(signal.RS232Signal(data=data, baud_rate=9600, duration=1.0))

                #self.environment.plot(['inputs'])
                self.send_inputs(inputs, outputs)
                #self.environment.plot(['outputs', str(list(reset.signal.changes())[-2][0]), str(reset.signal.duration)])

                if valid.signal.final_value == 1:
                    t1 = list(recorded_serial.signal.changes())[-1][0]
                    t2 = list(valid.signal.changes())[-1][0]

                    print(password + c, t2-t1)
                    durations.append((c, t2-t1))

                else:
                    password += c
                    print('Password is', password)
                    return

            best_char = max(durations, key=lambda d: d[1])[0]
            password += best_char
            print('Building password:', password)


class SerialTest(DeviceTest):
    test_name = 'Serial Glitching Test'

    parameters = [
    ]

    relevant_inputs = [
        TestIOMapping('Serial')
    ]
    relevant_outputs = [
        TestIOMapping('SerialLoopback')
    ]

    def run(self, inputs, outputs): 
        serialout = self.relevant_input_values[0]    
        serialloop = self.relevant_output_values[0]
 
        for c in string.digits:
            serialout.signal = signal.RS232Signal(data=c, baud_rate=9600, duration=0.1).resample_factor(10).glitch(15.5/9600, 0.1/9600)

            self.environment.plot(['inputs'])
            self.send_inputs(inputs, outputs)
            self.environment.plot(['outputs'])

class PICSingleInstructionGlitch(DeviceTest):
    test_name = 'PIC Single Instruction Glitch'

    parameters = [
        #TestParameter('Duration', ('s', 'cycles'), float)
    ]

    relevant_inputs = [
        TestIOMapping('Reset'),
        TestIOMapping('Button'),
        TestIOMapping('Clock')
    ]

    relevant_outputs = [
    ]

    def run(self, inputs, outputs):
        #Reminder: these are the zybo pins
        reset = self.relevant_input_values[0]
        button = self.relevant_input_values[1]
        clock = self.relevant_input_values[2]

        #clock_rate = 4000000
        clock_rate = 50000

        reset.signal = signal.Signal(initial_value=signal.LOW, sample_rate=2*clock_rate, duration=16/clock_rate)
        reset.signal = reset.signal.append(signal.Signal(initial_value=signal.HIGH, sample_rate=2*clock_rate, duration=16/clock_rate))


        button_high = 64/clock_rate #How long to hold the start button high
        button_low = 64/clock_rate   #How long to press the button
        #button_rest = 32/clock_rate #Leave the button high for the remainder
        button_rest = 64/clock_rate #Leave the button high for the remainder

        button.signal = signal.Signal(initial_value=signal.HIGH, sample_rate=2*clock_rate, duration=button_high)
        button.signal = button.signal.append(signal.Signal(initial_value=signal.LOW, sample_rate=2*clock_rate, duration=button_low))
        button.signal = button.signal.append(signal.Signal(initial_value=signal.HIGH, sample_rate=2*clock_rate, duration=button_rest))

        #Empirically measured that we should glitch at 20 cycles after the
        #button press
        #glitch_location=button_high+(20/clock_rate)
        #glitch_location_samples = int(glitch_location*clock.signal.sample_rate)
        #print('glitch_location (seconds): ', glitch_location)
        #print('glitch_location (samples): ', glitch_location_samples)

        sample_rate=40*clock_rate
        #glitch_start = 82.71
        #glitch_start = 84.00

        #for glitch_start in range(165,170):
        for glitch_start in [165]:

            #Manually generate a clock        
            clock.signal = signal.Signal(initial_value=signal.LOW, duration=0.5/clock_rate, sample_rate=sample_rate)
            for factor in [0.01]:
            #for factor in [0.01,0.025,0.05,0.10,0.20]:
                glitch_size=factor/clock_rate
                glitch_size=25e-08
                #glitch_size=5e-06
                print("Trying a glitch of %g at %g "%(glitch_size,glitch_start))
                for i in range(200):
                    if i % 2 == 0:
                        signal_value = signal.HIGH
                    else:
                        signal_value = signal.LOW

                    if i == glitch_start or i == glitch_start+1:
                        #Insert glitch
                        duration=glitch_size
                        print("Inserted glitch at time ", i*(0.5/clock_rate))
                    else:
                        duration=0.5/clock_rate
                    clock.signal = clock.signal.append(signal.Signal(initial_value=signal_value, sample_rate=sample_rate, duration=duration))

                #self.environment.plot(['inputs'])
                self.send_inputs(inputs, outputs)

                if not self.behavior_model.validate(inputs, outputs):
                    print("---")
                    print("Behavior model not validated")
                    print("glitch_start: ", glitch_start)
                    print("glitch_size: ", glitch_size)
                    print("---")


                    print("---")


class ClockGlitchDev(DeviceTest):
    test_name = 'Clock Glitching Development'

    parameters = [
        #TestParameter('Duration', ('s', 'cycles'), float)
    ]

    relevant_inputs = [
        TestIOMapping('Reset'),
        TestIOMapping('Manual Clock'),
        TestIOMapping('Clock') #Kept clock on 3 to line up with PIC test
    ]

    relevant_outputs = [
    ]

    def run(self, inputs, outputs):
        #Reminder: these are the zybo pins
        reset = self.relevant_input_values[0]
        manual_clock = self.relevant_input_values[1]
        clock = self.relevant_input_values[2]

        clock_rate = 50000
        #Give a signal so this ends
        reset.signal = signal.Signal(initial_value=signal.LOW, sample_rate=2*clock_rate, duration=16/clock_rate)
        reset.signal = reset.signal.append(signal.Signal(initial_value=signal.HIGH, sample_rate=2*clock_rate, duration=16/clock_rate))

        glitch_size=0.25/clock_rate
        #glitch_size=0.0050/
        glitch_start=10.0

        #Manually generate a clock        
        manual_clock.signal = signal.Signal(initial_value=signal.LOW, duration=0.5/clock_rate, sample_rate=4*clock_rate)
        for i in range(280):
            if i % 2 == 0:
                signal_value = signal.HIGH
            else:
                signal_value = signal.LOW

            if i == glitch_start:
                #Insert glitch
                duration=glitch_size
            else:
                duration=0.5/clock_rate
            manual_clock.signal = manual_clock.signal.append(signal.Signal(initial_value=signal_value, sample_rate=4*clock_rate, duration=duration))

        print("Trying a glitch of %g at %g "%(glitch_size,glitch_start))
        clock.signal = signal.Clock(clock_rate)
        clock.signal = clock.signal \
            .unroll(140) \
            .glitch(glitch_start/clock_rate, glitch_size)\

        #self.environment.plot(['inputs'])

        self.send_inputs(inputs, outputs)

        #self.environment.plot(['outputs'])

tests = [
    BasicPlayback, 
    ClockRateTest, 
    MaxGlitchDuration, 
    ResponseTimeTest, 
    MaxNumGlitchTest, 
    MaxDriftTest, 
    CanBusDeviceTest,
    ButtonPulseWidthTest, 
    PICSingleInstructionGlitch,
    SerialFuzzer, 
    SerialTest,
    ClockGlitchDev
    ]
