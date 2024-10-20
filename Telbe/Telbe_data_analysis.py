'''
Teddy Mercer
2024-09-07

The Telbe class is a class that is used to parse and process Thz data files at Telbe.

The class can parse both regular and LockIn files, extract metadata from the filename, calculate the FFT of the signal.

Functionality includes:
- Parsing regular and LockIn files
- Extracting metadata from the filename
- Calculating the FFT of the signal
- Adding an offset to a specified parameter
- Combining data from two Telbe objects
- Interpolating the Signal data to new time values
- Fitting a Gaussian to the data of specified attributes within a given range

Example usage:
```python
telbe = Telbe()
telbe.parse_file('data/2021-06-01_10-00-00_WG')
telbe.calc_fft()
```

The class is designed to be used in a Jupyter notebook or a Python script.

THINGS I NEED TO DO - As of September 7th, 2024: 
*   The fft function is a little buggy
*   Finding a way to auto detect keys based off common words or phrases
*   Interpolate isnt tested yet
*   Document, examples, README.md
'''

## imports:
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

class Telbe:
    def __init__(self):
        self.meta = {}
        self.Pixel = np.array([])
        self.Time = np.array([])
        self.Signal = np.array([])

    def parse_file(self, file_path, key_pattern=None):
        """
        Parses the input file and optionally extracts metadata from the filename
        based on a provided key pattern.

        Args:
            file_path (str): Path to the file or the file name to be parsed.
            key_pattern (str): A key pattern representing the structure of the filename.
                               Example: 'Num_Type_Element_Fluence_FluenceVal_BandpassFilterval_bandpass_HPFval_HPF'
        """
        file_name = file_path.split('/')[-1]
        words = file_name.split('_')

        if key_pattern:
            self._extract_meta_from_filename(words, key_pattern)

        if 'LockIn' in words:
            self._parse_lockin_file(file_path)
        else:
            self._parse_regular_file(file_path)

    def _extract_meta_from_filename(self, words, key_pattern):
        """
        Extracts metadata from the filename based on a key pattern and stores it in the meta dictionary.

        Args:
            words (list): The components of the filename split by '_'.
            key_pattern (str): A key pattern string that describes the structure of the filename.
        """
        keys = key_pattern.split('_')
        if len(keys) != len(words):
            raise ValueError("Filename components do not match the provided key pattern.")

        for key, word in zip(keys, words):
            self.meta[key] = word

        print(f"Extracted metadata: {self.meta}")


    def _parse_lockin_file(self, file_path):
        """Parses LockIn type files and processes Pixel, Time, and Signal."""
        try:
            data = np.loadtxt(file_path)
            self.Pixel = data[:, 2]
            self.Time = data[:, 0]
            self.Signal = data[:, 1] * 5e3

            #self.Time -= self.Time[0]
            self.Time *= 6.67 #twice because one step of the stage is twice delay of beam

            #idx_max = np.argmax(self.Signal[:120])
            #self.Time -= self.Time[idx_max]
            self.Time-=244.12
            # Extract meta data
            #self._extract_meta(file_path.split('_'), 'WG')
            print('This is a LockIn file.')
        except Exception as e:
            print(f"Error parsing LockIn file: {e}")

    def _parse_regular_file(self, file_path, tester = True):
        """Parses regular files for Pixel, Time, and Signal data."""
        data_section_started = False

        try:
            with open(file_path, 'r') as file:
                for line in file:
                    line = line.strip()
                    if not line:
                        continue

                    if not data_section_started and "Pixel value" in line:
                        data_section_started = True
                        continue

                    if data_section_started:
                        data_line = line.split()
                        if len(data_line) == 3:
                            try:
                                self.Pixel = np.append(self.Pixel, float(data_line[0]))
                                self.Time = np.append(self.Time, float(data_line[1]))
                                self.Signal = np.append(self.Signal, float(data_line[2]))
                            except ValueError as e:
                                print(f"Error parsing line '{line}': {e}")
                        else:
                            print(f"Skipping malformed line: '{line}'")
                    elif ":" in line:
                        key, value = line.split(":", 1)
                        self.meta[key.strip()] = value.strip()

            idx_max = np.argmax(self.Signal[10:500])
            if tester == False:
              self.Time = self.Time - self.Time[idx_max]
            else:
              self.Time = self.Time
        except Exception as e:
            print(f"Error parsing regular file: {e}")

    def _extract_meta(self, words, character):
        """Extracts meta information based on a character string."""
        matching_words = [word for word in words if character in word]
        if matching_words:
            self.meta[character] = matching_words[0]

    def to_dataframe(self):
        """Converts the parsed data into a pandas DataFrame.
        In case you like that sort of thing
        """
        return pd.DataFrame({
            'Pixel': self.Pixel,
            'Time, ps': self.Time,
            'Signal': self.Signal
        })

    def calc_fft(self, window=None):
        """Calculates the FFT of the signal, optionally within a window."""
        T = np.mean(np.diff(self.Time))
        sampling_rate = 1 / T

        data = self.Signal
        if window:
            idx_begin = np.searchsorted(self.Time, window[0])
            idx_end = np.searchsorted(self.Time, window[1], side='right') - 1
            print(idx_begin,idx_end)
            data = data[idx_begin:idx_end + 1]

        fft_values = np.fft.fft(data)
        frequencies = np.fft.fftfreq(len(data), T)
        magnitude = np.abs(fft_values)

        self.positive_frequencies = frequencies[:len(frequencies) // 2]
        self.positive_magnitude = magnitude[:len(magnitude) // 2]

        print('FFT calculated. Attributes: positive_frequencies, positive_magnitude.')

    def add_offset(self, param_name, offset):
        """Adds an offset to a specified parameter."""
        if hasattr(self, param_name):
            setattr(self, param_name, getattr(self, param_name) + offset)
            return self
        else:
            raise AttributeError(f"Parameter '{param_name}' does not exist in the object.")

    def combine_data(self, other, combine_attrs):
        """
        Combines specified attributes (like Time, Signal, Pixel) of two Telbe objects.
        Attributes must exist in both objects.

        Args:
            other (Telbe): Another Telbe object to combine with the current one.
            combine_attrs (list): List of attribute names to combine (e.g., ['Time', 'Signal']).

        Returns:
            Telbe: A new Telbe object containing the combined data for specified attributes.
        """
        combined = Telbe()

        for attr in combine_attrs:
            if hasattr(self, attr) and hasattr(other, attr):

                combined_data = np.concatenate((getattr(self, attr), getattr(other, attr)))

                if attr == 'Time':

                    sorted_indices = np.argsort(combined_data)


                    combined.Time = combined_data[sorted_indices]


                    for other_attr in combine_attrs:
                        if other_attr != 'Time':
                            combined_attr_data = np.concatenate((getattr(self, other_attr), getattr(other, other_attr)))
                            setattr(combined, other_attr, combined_attr_data[sorted_indices])
                else:

                    setattr(combined, attr, combined_data)
            else:
                raise AttributeError(f"Both Telbe objects must have the '{attr}' attribute.")
        return combined

    def interpolate_data(self, new_time, kind='linear'):
        """
        Interpolates the Signal data to new time values.

        Args:
            new_time (np.array): Array of new time values for interpolation.
            kind (str): Type of interpolation to use (default is 'linear').
                        Options include 'linear', 'nearest', 'zero', 'slinear', 'quadratic', 'cubic'.
                        --- never going to use really most of these

        Returns:
            np.array: Interpolated Signal values at the new time points.
        """
        if self.Time.size == 0 or self.Signal.size == 0:
            raise ValueError("Time and Signal data must be populated before interpolation.")

        interpolating_function = interp1d(self.Time, self.Signal, kind=kind, fill_value="extrapolate")
        interpolated_signal = interpolating_function(new_time)
        return interpolated_signal

    @staticmethod
    def _gaussian(x, height, center, width):
        """Defines a Gaussian function."""
        return height * np.exp(-((x - center) ** 2) / (2 * width ** 2))

    def fit_gaussian_to_range(self, x_attr, y_attr, x_range):
        """
        Fits a Gaussian to the data of specified attributes within the given range.

        Args:
            x_attr (str): The attribute name to use for x data (e.g., 'Time', 'Pixel').
            y_attr (str): The attribute name to use for y data (e.g., 'Signal', 'Pixel').
            x_range (tuple): A tuple specifying the (min_value, max_value) to fit the Gaussian.

        Returns:
            height (float): Height of the fitted Gaussian.
            fwhm (float): Full Width at Half Maximum of the Gaussian.
            center (float): Center position of the Gaussian.
        """
        if not hasattr(self, x_attr) or not hasattr(self, y_attr):
            raise AttributeError(f"One or both of the attributes '{x_attr}' or '{y_attr}' do not exist.")


        x_data = getattr(self, x_attr)
        y_data = getattr(self, y_attr)

        mask = (x_data >= x_range[0]) & (x_data <= x_range[1])
        x_data = x_data[mask]
        y_data = y_data[mask]

        initial_guess = [np.max(y_data), x_data[np.argmax(y_data)], np.std(x_data)]

        try:
            popt, _ = curve_fit(self._gaussian, x_data, y_data, p0=initial_guess)

            height, center, width = popt

            fwhm = 2 * np.sqrt(2 * np.log(2)) * abs(width)

            return height, fwhm, center
        except RuntimeError as e:
            print(f"Error fitting Gaussian: {e}")
            return None, None, None