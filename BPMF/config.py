import os
import inspect
import pandas as pd


class Config:
    def __init__(self, parameters):
        for attr in [
            "MIN_FREQ_HZ",
            "MAX_FREQ_HZ",
            "SAMPLING_RATE_HZ",
            "TEMPLATE_LEN_SEC",
            "SEARCH_WIN",
            "MATCHED_FILTER_STEP_SAMP",
            "N_DEV_MF_THRESHOLD",
            "N_DEV_BP_THRESHOLD",
            "DATA_BUFFER_SEC",
        ]:
            # allow meaningless default values in case BPMF is used for
            # very specific usage
            parameters.setdefault(attr, -10)
        for attr in [
            "INPUT_PATH",
            "NETWORK_PATH",
            "MOVEOUTS_PATH",
            "OUTPUT_PATH",
            "NLLOC_INPUT_PATH",
            "NLLOC_OUTPUT_PATH",
            "NLLOC_BASENAME",
        ]:
            parameters.setdefault(attr, "")
        for key in parameters:
            setattr(self, key, parameters[key])
        # for backward compatibility
        self.dbpath = self.INPUT_PATH
        self.PACKAGE = os.path.dirname(inspect.getfile(inspect.currentframe())) + '/'


if not os.path.isfile("BPMF_parameters.csv"):
    print("Could not find the BPMF_parameters.csv file in "
          "current working directory.")
    parameters = {}
else:
    parameters = pd.read_csv("BPMF_parameters.csv",
            index_col=0).to_dict()["parameter_value"]

cfg = Config(parameters)
