import os
import inspect

str_parameters = [
            "INPUT_PATH",
            "NETWORK_PATH",
            "MOVEOUTS_PATH",
            "OUTPUT_PATH",
            "NLLOC_INPUT_PATH",
            "NLLOC_OUTPUT_PATH",
            "NLLOC_BASENAME",
            "PARAMETER_FILE",
        ]
float_parameters = [
            "MIN_FREQ_HZ",
            "MAX_FREQ_HZ",
            "SAMPLING_RATE_HZ",
            "TEMPLATE_LEN_SEC",
            "N_DEV_MF_THRESHOLD",
            "N_DEV_BP_THRESHOLD",
            "DATA_BUFFER_SEC",
            "BUFFER_EXTRACTED_EVENTS_SEC",
        ]
int_parameters = [
            "SEARCH_WIN",
            "MATCHED_FILTER_STEP_SAMP",
        ]

class Config:
    def __init__(self, parameters):
        for attr in int_parameters:
            # allow meaningless default values in case BPMF is used for
            # very specific usage
            parameters.setdefault(attr, -10)
        for attr in float_parameters:
            parameters.setdefault(attr, -10.)
        for attr in str_parameters:
            parameters.setdefault(attr, "")
        for key in parameters:
            setattr(self, key, parameters[key])
        # for backward compatibility
        self.dbpath = self.OUTPUT_PATH
        self.PACKAGE = os.path.dirname(inspect.getfile(inspect.currentframe())) + '/'

parameter_types = {}
for param in str_parameters:
    parameter_types[param] = str
for param in float_parameters:
    parameter_types[param] = float
for param in int_parameters:
    parameter_types[param] = int

if not os.path.isfile("BPMF_parameters.cfg"):
    print("Could not find the BPMF_parameters.cfg file in "
          "current working directory.")
    parameters = {}
else:
    parameters = {}
    with open("BPMF_parameters.cfg", "r") as fparams:
        for line in fparams:
            key, value = line.split("=")
            key, value = key.strip(), value.strip()
            parameters[key] = parameter_types[key](value)

cfg = Config(parameters)
