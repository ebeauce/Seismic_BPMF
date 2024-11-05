# tests for remove_response functions in utils.py

import pytest
import numpy as np

import obspy

from BPMF.utils import remove_response, attach_response, get_response

# what to test
# remove_response with a trace and inventory
# remove_response with a trace and attached response
# attach_response
# multiple things to test for get_response

def test_remove_response_with_inv():
    tr = obspy.read("./stream.mseed")[0]
    inv = obspy.read_inventory("./station.xml")
    old_data = tr.data
    remove_response(tr, inv)
    assert not np.all(tr.data == old_data)

def test_attach_response_list_inv():
    tr = obspy.read("stream.mseed")[0]
    inv = obspy.read_inventory("station.xml")
    attach_response(tr, [inv])
    assert hasattr(tr.stats, "response")

def test_attach_response_single_inv():
    tr = obspy.read("stream.mseed")[0]
    inv = obspy.read_inventory("station.xml")
    attach_response(tr, inv)
    assert hasattr(tr.stats, "response")

def test_remove_response_attached_response():
    tr = obspy.read("stream.mseed")[0]
    inv = obspy.read_inventory("station.xml")
    attach_response(tr, inv)
    old_data = tr.data
    remove_response(tr)
    assert not np.all(tr.data == old_data)

# testing get_response
# branches to test:
# - inventories is None
#   - catch ValueError
# - inventories is None and 'response' in trace.stats
#   - not isinstance(trace.stats.response, Response): catch TypeError
#   - else assert return is type Response
# - isinstance(inventories, Inventory) or isinstance(inventories, Network): assert return is type Response
# - isinstance(inventories, str): assert return is type Response
#   - read in station.xml

def test_get_response_with_inv():
    # Trace with inventory
    tr = obspy.read("stream.mseed")[0]
    inv = obspy.read_inventory("station.xml")
    assert type(get_response(tr, inv)) == obspy.core.inventory.response.Response

def test_get_response_with_net():
    # Trace with Network
    tr = obspy.read("stream.mseed")[0]
    net = obspy.read_inventory("station.xml").networks[0]
    assert type(get_response(tr, net)) == obspy.core.inventory.response.Response

def test_get_response_from_file():
    # Trace with filename for inventory
    tr = obspy.read("stream.mseed")[0]
    assert type(get_response(tr, "station.xml")) == obspy.core.inventory.response.Response

def test_get_response_inv_None_resp_attached():
    # Trace with response attached already
    tr = obspy.read("stream.mseed")[0]
    inv = obspy.read_inventory("station.xml")
    attach_response(tr, inv)
    assert type(get_response(tr)) == obspy.core.inventory.response.Response

def test_get_response_inv_None():
    # Errors when given a trace with no response attached and nop inventory
    tr = obspy.read("stream.mseed")[0]
    try:
        get_response(tr)
    except Exception as e:
        assert type(e) == ValueError

def test_get_response_inv_None_bad_resp_attached():
    # Erros when the response attached to the trace is not of type Response
    tr = obspy.read("stream.mseed")[0]
    tr.stats.response = 10
    try:
        get_response(tr)
    except Exception as e:
        assert type(e) == TypeError