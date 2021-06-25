import os
import sys
import inspect
import numpy as np


class Config():
    def __init__(self, parameters):
        self.__dict__ = parameters
        self.base = os.getcwd()
        self.data = os.path.join(self.base, self.__dict__['input_path'])
        self.network_path = os.path.join(self.base, self.__dict__['network_path'])
        self.moveouts_path = os.path.join(self.base, self.__dict__['moveouts_path'])
        self.dbpath = self.chk_trailing(self.__dict__['output_path'])
        self.package = os.path.dirname(inspect.getfile(inspect.currentframe()))

    def chk_folder(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

    def chk_trailing(self, to_test):
        if to_test.endswith('/'):
            return to_test
        else:
            return to_test + '/'

package = os.path.dirname(inspect.getfile(inspect.currentframe())) + '/'
# read in study parameters
parameters_path = os.path.join(os.getcwd(), 'parameters.cfg')
if os.path.isfile(parameters_path):
    with open(parameters_path) as file:
        param_dict = {}
        for line in file:
            key, value = line.split('=')
            key = key.strip()
            value = value.strip()
            try:
                value = np.float32(value)
            except Exception:
                pass
            if isinstance(value, str) and len(value.split(',')) > 1:
                value = [np.float32(freq) for freq in value.split(',')]
            param_dict[key] = value
    cfg = Config(param_dict)
else:
    print(f'Couldn\'t find the parameter file at {parameters_path}.')
    print('See https://github.com/ebeauce/Seismic_BPMF/ for an example'
          ' of a parameters.cfg file.')
    sys.exit()

## specific parameter config
## frequency bands
#if 'min_freq' and 'max_freq' in param_dict:
#    param_dict['freq_bands'] = []
#    if isinstance(param_dict['min_freq'], list):
#        n_freqs = len(param_dict['min_freq'])
#        min_freq = param_dict.pop('min_freq')
#        max_freq = param_dict.pop('max_freq')
#        for f in range(n_freqs):
#            param_dict['freq_bands'].append([min_freq[f], max_freq[f]])
#    else:
#        param_dict['freq_bands'].append(
#            [param_dict['min_freq'], param_dict['max_freq']])
#
## sampling_rate (needs to be an int)
#if 'sampling_rate' in param_dict:
#    param_dict['sampling_rate'] = np.int32(param_dict['sampling_rate'])

