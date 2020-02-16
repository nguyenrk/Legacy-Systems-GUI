import csv
import platform
import pickle
import os
import struct
import tempfile
import time

from collections import defaultdict

import matplotlib.pyplot as plt
import saleae

from .utils import *
from . import exceptions

if 'Windows' == platform.system():
    from pywinauto.findwindows import find_window

# Constants
LOW  = 0
HIGH = 1
MHZ = 1000 * 1000

class RecordingSession:
    @staticmethod
    def sample_rates(channels):
        sal = saleae.Saleae()
        sal.set_active_channels(digital=channels, analog=None)
        return [r[0] for r in sal.get_all_sample_rates()]

    def __init__(self, sample_rate=10*MHZ, max_duration=600, outputs=None):
        channels = [o.index-1 for o in outputs]
        self.channels = channels
        self.sample_rate = sample_rate
        
        self.sal = saleae.Saleae()

        self.sal.set_active_channels(digital=channels, analog=None)
        self.sal.set_sample_rate((sample_rate, 0))
        self.sal.set_capture_seconds(max_duration)

        for output in outputs:
            self.sal.set_trigger_one_channel(output.index-1, output.trigger)

        self.start_time = time.monotonic()
        self.sal.capture_start()

        print("Sleeping for 3 seconds...")
        time.sleep(3)
        """
        time.sleep(0.5)
        # Wait for the capture to actually start to ensure we don't miss the first part of our data
        if 'Windows' == platform.system():
            while 1:
                try:
                    find_window(title='Logic')
                    return
                except:
                    pass
        else:
            # TODO: improve Linux support here, actually look for a window
            time.sleep(3)
        """

    def wait_until_finished(self):
        while not self.sal.is_processing_complete():
            time.sleep(0.1)

    def finish_recording(self):
        time.sleep(0.5)
        result = self.sal.capture_stop()
        time.sleep(0.1)

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'record.bin')
            
            # TODO: benchmark whether each_sample would be faster when signals
            # are high-speed clocks (definitely worse with high sample rate but slow signals, though)
            export_settings = {
                'format': 'binary',
                'each_sample': False,
                'word_size': 8,
                'no_shift': False
            }

            print('Exporting Saleae data...')
            self.sal.export_data2(path, **export_settings)

            signals = Signal.load_from_saleae_recording(
                export_settings,
                path,
                channel_count=len(self.channels), 
                sample_rate=self.sample_rate, 
                est_duration=time.monotonic() - self.start_time)
        
            os.remove(path)

        self.sal.close_all_tabs()

        return signals

class Signal:
    """A representation of a recorded or constructed signal."""
    def __init__(self, sample_rate=20*MHZ, initial_value=LOW, duration=0):
        self.sample_rate = round(sample_rate)
        self.initial_value = initial_value

        if initial_value not in (LOW, HIGH):
            raise SignalException('Invalid initial value: {}'.format(initial_value))

        self.sample_change_indices = []
        self.sample_count = round(duration * self.sample_rate)

        self.repeat = False

    # Alternate constructors
    @classmethod
    def from_samples(self, sample_rate, samples):
        signal = Signal(sample_rate=sample_rate, duration=(len(samples) + 1)/sample_rate, initial_value=samples[0])

        for i in range(1, len(samples)):
            if samples[i] != samples[i-1]:
                signal.sample_change_indices.append(i)

        return signal

    @classmethod
    def start_recording(cls, *args, **kwargs):
        return RecordingSession(*args, **kwargs)

    @classmethod
    def load_from_saleae_recording_csv(cls, filename, channel_count, sample_rate, est_duration=None):
        signals = [Signal(sample_rate=sample_rate) for _ in range(channel_count)]
        percent_complete = 0

        row_count = 0
        with open(filename) as data_file:
            reader = csv.reader(data_file)
            for i, row in enumerate(reader):
                
                if i == 0:
                    # Skip header line
                    continue

                if i == 1:
                    start_time = round(float(row[0]) * sample_rate)

                    # Initialize signals
                    for channel, bit in enumerate(get_bits(row[1], channel_count)):
                        signals[channel].initial_value = bit

                else:
                    sample_index = round(float(row[0]) * sample_rate) - start_time
                    for channel, bit in enumerate(get_bits(row[1], channel_count)):
                        if signals[channel].final_value != bit:
                            signals[channel].sample_change_indices.append(sample_index)

                    if i % 1024 == 0:  # for efficiency, only check occasionally
                        current_percent_complete = (sample_index / sample_rate) / est_duration
                        if current_percent_complete - percent_complete > 0.0001:
                            percent_complete = current_percent_complete
                            print('{:.2%}'.format(percent_complete))

            for channel in range(channel_count):
                signals[channel].sample_count = 1 + round(float(row[0]) * sample_rate - start_time)

        return signals

    @classmethod
    def load_from_saleae_recording_binary_diff(cls, filename, channel_count, sample_rate, est_duration=None):
        signals = [Signal(sample_rate=sample_rate) for _ in range(channel_count)]
        last_vals = [0, 0, 0, 0, 0, 0, 0, 0]

        percent_complete = 0
        t = time.monotonic()
        fsize = os.stat(filename).st_size
        #print('{}, {:0.2f} MB'.format(filename, fsize / (1024**2)))

        lnfmt = 'QB'
        lnsz = struct.calcsize(lnfmt)

        with open(filename, 'rb') as data_file:
            # Initialize signals
            data = data_file.read(lnsz)
            index, val = struct.unpack(lnfmt, data)

            for i, s in enumerate(signals):
                s.initial_value = val & 1
                last_vals[i] = val & 1
                val = val >> 1

            # Parse remaining samples
            bytes_read = 0
            while True:
                data = data_file.read(lnsz)
                if not data:
                    break
                bytes_read += len(data)
                index, val = struct.unpack(lnfmt, data)

                for i, s in enumerate(signals):
                    if (val & 1) != last_vals[i]:
                        last_vals[i] = val & 1
                        s.sample_change_indices.append(index)
                    val = val >> 1

                p = bytes_read / fsize 
                if p - percent_complete > 0.01:
                    percent_complete = p
                    print('{:.0%} ({:.02f} sec)'.format(p, time.monotonic() - t), end='\r')

        print('{:.0%} ({:.02f} sec)'.format(1, time.monotonic() - t))

        for s in signals:
            s.sample_count = index + 1

        return signals

    @classmethod
    def load_from_saleae_recording_binary(cls, filename, channel_count, sample_rate, est_duration=None):
        # The differential mode is in every case I tested much faster than this, even when the signal
        # rate approaches the sample rate (despite this  one producing *much* smaller files, since it
        # drops the 64-bit index on every sample)
        signals = [Signal(sample_rate=sample_rate) for _ in range(channel_count)]
        last_vals = [0, 0, 0, 0, 0, 0, 0, 0]

        percent_complete = 0
        t = time.monotonic()
        fsize = os.stat(filename).st_size
        #print('{}, {:0.2f} MB'.format(filename, fsize / (1024**2)))

        lnfmt = 'B'
        lnsz = struct.calcsize(lnfmt)

        with open(filename, 'rb') as data_file:
            # Initialize signals
            data = data_file.read(lnsz)
            val, = struct.unpack(lnfmt, data)

            for i, s in enumerate(signals):
                s.initial_value = val & 1
                last_vals[i] = val & 1
                val = val >> 1

            # Parse remaining samples
            bytes_read = 0
            index = 1
            while True:
                data = data_file.read(lnsz)
                if not data:
                    break
                bytes_read += len(data)
                val, = struct.unpack(lnfmt, data)

                for i, s in enumerate(signals):
                    if (val & 1) != last_vals[i]:
                        last_vals[i] = val & 1
                        s.sample_change_indices.append(index)
                    val = val >> 1

                index += 1

                p = bytes_read / fsize 
                if p - percent_complete > 0.01:
                    percent_complete = p
                    print('{:.0%} ({:.02f} sec)'.format(p, time.monotonic() - t), end='\r')

        print('{:.0%} ({:.02f} sec)'.format(1, time.monotonic() - t))
        return signals

    @classmethod
    def load_from_saleae_recording(cls, export_settings, *args, **kwargs):
        print('Loading data from Saleae...')
        if export_settings['format'] == 'binary' \
                and export_settings['each_sample'] is False \
                and export_settings['no_shift'] is False \
                and export_settings['word_size'] == 8:
            return cls.load_from_saleae_recording_binary_diff(*args, **kwargs)
        elif export_settings['format'] == 'binary' \
                and export_settings['each_sample'] is True \
                and export_settings['no_shift'] is False \
                and export_settings['word_size'] == 8:
            return cls.load_from_saleae_recording_binary(*args, **kwargs)
        elif export_settings['format'] == 'csv':
            # TODO: which exact settings does our CSV method support?
            return cls.load_from_saleae_recording_csv(*args, **kwargs)
        else:
            raise NotImplementedError('The {} Saleae export format is not implemented!'.format(
                export_settings['format']))

    @classmethod
    def record(cls, *args, **kwargs):
        session = RecordingSession(*args, **kwargs)
        session.wait_until_finished()
        return session.finish_recording()

    @classmethod
    def load(self, filename):
        with open(filename, 'rb') as f:
            return pickle.load(f)

    def __len__(self):
        return self.sample_count

    def __repr__(self):
        rowfmt = '{:06f} | {}\n'
        value = self.initial_value
        s = rowfmt.format(0, value)

        for index in self.sample_change_indices:
            value = 1 - value
            s += rowfmt.format(index / self.sample_rate, value)    

        return s

    def __str__(self):
        return 'duration {} sec, sampled at {}'.format(
                self.sample_count / self.sample_rate, display_rate(self.sample_rate))

    def __eq__(self, other):
        if not isinstance(other, Signal):
            return False

        if self.sample_count != other.sample_count:
            return False

        if self.sample_rate != other.sample_rate:
            return False

        if self.initial_value != other.initial_value:
            return False

        if self.sample_change_indices != other.sample_change_indices:
            return False

        return True

    def samples(self, length=None):
        """Generate samples"""
        # Replay samples
        count = 0
        
        while 1:
            index = 0
            value = self.initial_value
            for i in range(self.sample_count):
                if index < len(self.sample_change_indices) and self.sample_change_indices[index] == i:
                    value = 1 - value
                    index += 1

                yield value
                count += 1

                if length is not None and count == length:
                    break

            if not self.repeat:
                break

    def changes(self):
        value = self.initial_value
        yield (0, value)

        for index in self.sample_change_indices:
            value = 1 - value
            yield (index / self.sample_rate, value)

    def edges(self, mode='all'):
        """mode can be 'all', 'rising', 'falling'"""
        if mode == 'all':
            yield from (t for t, _ in self.changes())

        elif mode == 'rising':
            value = self.initial_value
            for index in self.sample_change_indices:
                value = 1 - value
                if value:
                    yield index / self.sample_rate

        elif mode == 'falling':
            value = self.initial_value
            for index in self.sample_change_indices:
                value = 1 - value
                if value == 0:
                    yield index / self.sample_rate

    def value_at_time(self, time):
        index = time * self.sample_rate
        count = len([i for i in self.sample_change_indices if i < index])

        if count % 2 == 0:
            return self.initial_value
        else:
            return 1 - self.initial_value
            
    def plot(self, show=True, xlim=None, highlight_groups=False, show_xlabel=True, **kwargs):
        x_values = [0]
        y_values = [self.initial_value]

        value = self.initial_value
        
        if self.repeat and xlim is None:
            xlim = [0, self.period]

        x_offset = 0
        while 1:
            for index in self.sample_change_indices:
                # Create a dummy sample just before change (not even at a sample location)
                # in order to create a nice, square waveform display

                t = x_offset + index / self.sample_rate
                if xlim is not None:
                    if t < xlim[0] or t > xlim[1]:
                        value = 1 - value
                        continue

                x_values.append(t)
                y_values.append(value)

                value = 1 - value
                x_values.append(t)
                y_values.append(value)

            if not self.repeat:
                break
            else:
                x_offset += self.duration
                if self.initial_value != self.final_value:
                    x_values.append(x_offset - 0.001 / self.sample_rate)
                    y_values.append(value)                    
                    value = 1 - value
                    x_values.append(x_offset)
                    y_values.append(value)

                if x_values[-1] > xlim[1]:
                    break

        x_values.append(x_offset + (self.sample_count - 1) / self.sample_rate)
        y_values.append(value)

        plt.plot(x_values, y_values)

        if highlight_groups:
            groups = self.find_groups(**kwargs)
            for a, b in groups:
                plt.axvspan(a, b, color='red', alpha=0.5)

        if show_xlabel:
            plt.xlabel('Time (s)')

        if xlim is not None:
            plt.xlim(xlim)

        plt.ylim([-0.1, 1.1])
        #plt.xticks([])
        plt.yticks([])

        if show:
            plt.show()

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @property
    def final_value(self):
        return self.initial_value if len(self.sample_change_indices) % 2 == 0 else 1 - self.initial_value

    @property
    def duration(self):
        return self.sample_count / self.sample_rate

    # Analysis
    def get_max_rate(self):
        """Find the fastest rates of change in the signal"""
        # Ignore the first duration here, we may have started recording in the middle of a level
        min_diff = self.sample_change_indices[1] - self.sample_change_indices[0]
        last_index = self.sample_change_indices[1]
        
        for index in self.sample_change_indices[2:]:
            diff  = (index - last_index)
            if diff < min_diff:
                min_diff = diff

            last_index = index

        return self.sample_rate / min_diff

    def guess_clock_rate(self):
        """Guess the "clock rate" of the signal

        This is intended to be used on a signal that is (or is suspected to be)
        a serial stream of digital data. 

        This is a home-grown algorithm.

        It aggregates times between changes in the signal. These times get 
        sorted into buckets. Times are put in the same bucket if they appear to
        either be the the "same" time, or multiples of that time.

        When a "multiple" of a time is added to a bucket, it gets divided into 
        several times. For example, if a time appears to be three times the 
        nominal bucket value, the time is divided by three, and then added 
        three times. This ensures that taking the average of the items in the 
        bucket produces the approximate time delta that the bucket is meant
        to represent. 

        An arbitrarily-chosen threshold of 5% is currently used to decide if a 
        time belongs in a bucket.

        Finally, the bucket with the most items is selected as the bucket used
        to calculate the guessed clock rate. 
        """
        if len(self.sample_change_indices) < 3:
            return 0

        buckets = defaultdict(list)
        min_diff = self.sample_change_indices[1] - self.sample_change_indices[0]
        last_index = self.sample_change_indices[1]

        for index in self.sample_change_indices[2:]:
            diff  = (index - last_index)
            buckets[diff].append(diff)

            if diff < min_diff:
                min_diff = diff

            last_index = index

        # Aggregate "close" buckets
        buckets = sorted(list(buckets.items()), reverse=True)
        aggregated_buckets = []

        while 1:
            if len(buckets):
                diff, items = buckets.pop()
            else:
                break

            for other_diff, other_items in aggregated_buckets:
                # Check if integer multiple of other diff 
                if (diff / other_diff) % 1 < 0.05:
                    count = round(diff / other_diff)
                    other_items.extend((i / count for i in items))
                    break

            else:
                aggregated_buckets.append((diff, items))

        # Recalculate first argument as average of items in bucket
        buckets = [(sum(b[1]) / len(b[1]), b[1]) for b in aggregated_buckets]

        # Find best bucket
        max_count = len(buckets[0][1])
        best_diff = buckets[0][0]

        for diff, items in buckets:
            if len(items) > max_count:
                max_count = len(items)
                best_diff = diff

        return self.sample_rate / best_diff

    def find_groups(self, by_synchronous=True, by_idle=None):
        clock_diff = self.sample_rate / self.guess_clock_rate()

        groups = []
        last_index = 0
        start = 0
        count = 0
        
        for index in self.sample_change_indices + [self.sample_count - 1]:
            diff  = (index - last_index)

            mod = (diff / clock_diff) % 1
            error = min(mod, 1 - mod)

            if (by_synchronous and error > 0.05) or (by_idle and diff > by_idle * clock_diff):
                if count > 0:
                    groups.append((start / self.sample_rate, last_index / self.sample_rate))

                # Reset group start when we get non-conforming diff
                start = index
                count = 0

            else:
                count += round(diff / clock_diff)

            last_index = index

        # Add last group
        if start != index:
            groups.append((start / self.sample_rate, index / self.sample_rate))

        return groups

    def get_first_change_time(self):
        return self.sample_change_indices[0] / self.sample_rate

    def get_last_change_time(self):
        return self.sample_change_indices[-1] / self.sample_rate


    # Mutations - return a new signal with the signal mutation applied
    def clone(self, transform=lambda i: i):
        """Return a copy of the signal"""
        signal = Signal(sample_rate=self.sample_rate, initial_value=self.initial_value)
        for index in self.sample_change_indices:
            signal.sample_change_indices.append(transform(index))

        signal.sample_count = transform(self.sample_count)
        return signal

    def truncate(self, start_time, end_time):
        initial_value = self.value_at_time(start_time)
        start_index = round(start_time * self.sample_rate)
        end_index = round(end_time * self.sample_rate)

        if self.repeat:
            # TODO: Unroll repeated signals enough to fill the requested region
            raise NotImplementedError('Truncating repeated signals is not currently supported')

        signal = self.clone(transform=lambda i: i-start_index)
        signal.initial_value = initial_value
        signal.sample_count = end_index - start_index
        signal.sample_change_indicies = [i for i in signal.sample_change_indices if i <= end_index]

        return signal
        
    def scale_x(self, multiplier):
        """Return a new signal"""
        return self.clone(lambda i: round(i * multiplier))
    
    def insert_delay(self, delay_duration, delay_start_time=0):
        """Insert a delay into a signal"""
        delay_samples = round(self.sample_rate * delay_duration)
        return self.clone(lambda i: i + delay_samples if delay_start_time == 0 or i >= round(self.sample_rate / delay_start_time) else i)

    def append(self, other):
        """Append a signal to a clone of this one"""
        signal = self.clone()

        if self.sample_rate == 0: # Sample rate of zero is a "null signal"
            signal.sample_rate = other.sample_rate
            signal.initial_value = other.initial_value

        if signal.sample_rate != other.sample_rate:
            raise exceptions.SignalIncompatibleException('Cannot append signals with different sample rates')
        
        if signal.final_value != other.initial_value:
            signal.sample_change_indices.append(signal.sample_count)

        for index in other.sample_change_indices:
            signal.sample_change_indices.append(signal.sample_count + index)

        signal.sample_count += other.sample_count
        return signal

    def glitch(self, start, length, resample=False):
        if resample:
            s = self.resample_factor(resample)
        else:
            s = self

        start_index = round(start * s.sample_rate)
        end_index = round((start + length) * s.sample_rate)

        for new_index in (start_index, end_index):
            for j, index in enumerate(s.sample_change_indices):
                if new_index < index:
                    s.sample_change_indices.insert(j, new_index)
                    break

                elif new_index == index:
                    s.sample_change_indices.pop(j)
                    break
            else:
                s.sample_change_indices.append(new_index)

        return s

    def resample_factor(self, factor):
        s = self.scale_x(factor)
        s.sample_rate = self.sample_rate * factor
        return s


class Clock(Signal):
    # TODO 1: Force clocks to always use 2X their clock rate as sample rate if 50% duty cycle
    # for efficiency, ensure playback is compatible with unsynchronized rates

    # TODO 2: Restructure clock internals as Python generator type so it has infinite duration
    # and doesn't take up memory

    def __init__(self, clock_rate=MHZ, duty_cycle=0.50, sample_rate=None, initial_value=HIGH):
        period = 1 / clock_rate

        if sample_rate is None:
            sample_rate = round(clock_rate / duty_cycle)

        super().__init__(sample_rate=sample_rate, duration=period, initial_value=initial_value)
        self.clock_rate = clock_rate
        self.duty_cycle = duty_cycle
        self.repeat = True

        self.__build_clock()

    def __build_clock(self):
       """Generate the sample_change_indices based on current parameters

       Internal function
       """

       if self.duty_cycle >= 1.00 or self.duty_cycle <= 0.00:
           raise exceptions.ClockInvalidDutyCycle(self.duty_cycle)

       front_samples_per_change = round(self.sample_rate * self.duty_cycle / self.clock_rate)
       back_samples_per_change = round(self.sample_rate * (1 - self.duty_cycle) / self.clock_rate)
       samples_per_period = front_samples_per_change + back_samples_per_change

       if front_samples_per_change == 0 or back_samples_per_change == 0:
           # TODO: could be due to extreme duty cycle getting rounded to 0 for piece
           raise exceptions.ClockTooFast(self.clock_rate, self.sample_rate)

       # TODO: warning for duty cycle "rounding", e.g., 20% duty cycle with sample rate 
       # 4X clock rate gets rounded to 25%

       for i in range(0, round(self.sample_count / samples_per_period)):
           if i != 0:
               self.sample_change_indices.append(samples_per_period * i)
           self.sample_change_indices.append(samples_per_period * i + front_samples_per_change)

    @property
    def period(self):
        return 1 / self.clock_rate

    def __str__(self):
        return '{} clock{}'.format(
                display_rate(self.clock_rate),
                ', duty cycle: {:.02f}'.format(self.duty_cycle) if self.duty_cycle != 0.50 else '')

    @Signal.duration.setter
    def duration(self, value):
        """Rebuild the clock with a new duration"""
        self.sample_count = round(value * self.sample_rate)
        self.sample_change_indices = []
        self.__build_clock()

    def unroll(self, count):
        """Unroll the repeat signal into a non-repeat signal"""
        s = self
        for i in range(count-1):
            s = s.append(self)

        s.repeat = False
        return s

def Pulse(pulse_value=HIGH, pre_duration=1, duration=1, post_duration=1, sample_rate=20*MHZ):
    s = Signal(initial_value=1-pulse_value, duration=pre_duration, sample_rate=sample_rate)
    s = s.append(Signal(initial_value=pulse_value, duration=duration, sample_rate=sample_rate))
    s = s.append(Signal(initial_value=1-pulse_value, duration=post_duration, sample_rate=sample_rate))
    return s


class RS232Signal(Signal):
    def __init__(self, data=b'', baud_rate=115200, char_spacing=0, sample_rate=None, duration=None):
        if sample_rate is None:
            sample_rate = baud_rate

        super().__init__(sample_rate=sample_rate, initial_value=HIGH, duration=10/baud_rate)
        # Always start with a short idle time - one bit
        #self.sample_count = round(1 / baud_rate)
        self.baud_rate = baud_rate
        self.char_spacing = char_spacing
        self.data = data

        for c in data:
            self.append_char(c)

        if duration is not None and self.duration < duration:
            self.sample_count += round(duration * self.sample_rate)


    def append_char(self, char):
        start_time = self.sample_count * self.sample_rate

        # Flip to low for start bit
        self.sample_change_indices.append(self.sample_count)

        # Character
        last_bit = 0
        for i in range(8):
            bit = (ord(char) >> i) & 0x1

            change_sample_index = round(self.sample_count + (1 + i) * self.sample_rate / self.baud_rate)
            if last_bit != bit:
                self.sample_change_indices.append(change_sample_index)

            last_bit = bit

        # Stop bit
        if last_bit != 1:
            self.sample_change_indices.append(round(self.sample_count + 9 * self.sample_rate / self.baud_rate))

        self.sample_count += round(11 * self.sample_rate / self.baud_rate)

        # Add character spacing
        self.sample_count += round(self.char_spacing * self.sample_rate)

    def __str__(self):
        return 'RS232, {} baud, data:  {}, sampled at {}'.format(
                self.baud_rate, 
                repr(self.data),
                display_rate(self.sample_rate))

class CANSignal(Signal):
    def __init__(self, data=b'', baud_rate=1000000, char_spacing=0, sample_rate=None, duration=None):
        if sample_rate is None:
            sample_rate = baud_rate

        super().__init__(sample_rate=sample_rate, initial_value=HIGH, duration=10/baud_rate)
        # Always start with a short idle time - one bit
        #self.sample_count = round(1 / baud_rate)
        self.baud_rate = baud_rate
        self.char_spacing = char_spacing
        self.data = data

        for c in data:
            self.append_char(c)

        if duration is not None and self.duration < duration:
            self.sample_count += round(duration * self.sample_rate)


    def append_char(self, char):
        start_time = self.sample_count * self.sample_rate

        # Flip to low for start bit
        self.sample_change_indices.append(self.sample_count)

        # Character
        last_bit = 0
        for i in range(8):
            bit = (ord(char) >> i) & 0x1

            change_sample_index = round(self.sample_count + (1 + i) * self.sample_rate / self.baud_rate)
            if last_bit != bit:
                self.sample_change_indices.append(change_sample_index)

            last_bit = bit

        # Stop bit
        if last_bit != 1:
            self.sample_change_indices.append(round(self.sample_count + 9 * self.sample_rate / self.baud_rate))

        self.sample_count += round(11 * self.sample_rate / self.baud_rate)

        # Add character spacing
        self.sample_count += round(self.char_spacing * self.sample_rate)

    def __str__(self):
        return 'CAN, {} baud, data:  {}, sampled at {}'.format(
                self.baud_rate, 
                repr(self.data),
                display_rate(self.sample_rate))
