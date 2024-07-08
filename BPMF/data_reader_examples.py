import numpy as np


def data_reader_template(
    where,
    network="*",
    station="*",
    channel="*",
    location="*",
    starttime="*",
    endtime="*",
    **kwargs,
):
    """Data reader for BPMF.

    Any data reader must have the present signature.

    Parameters
    -----------
    where: string
        Path to data file or root data folder.
    network: string or list, optional
        Code(s) of the target network(s).
    station: string or list, optional
        Code(s) of the target station(s).
    channel: string or list, optional
        Code(s) of the target channel(s).
    location: string or list, optional
        Code(s) of the target location(s).
    starttime: string or obspy.UTCDateTime, optional
        Target start time.
    endtime: string or obspy.UTCDateTime, optional
        Target end time.

    Returns
    -------
    traces: obspy.Stream
        The seismic data.
    """
    from obspy import Stream

    traces = Stream()
    # read your data into traces
    return traces


def data_reader_pyasdf(
    where,
    network="*",
    station="*",
    channel="*",
    location="*",
    starttime="*",
    endtime="*",
    tag="raw",
    **kwargs,
):
    """Data reader for BPMF based on the ASDF format.

    Parameters
    -----------
    where: string
        Path to data file or root data folder.
    network: string or list, optional
        Code(s) of the target network(s).
    station: string or list, optional
        Code(s) of the target station(s).
    channel: string or list, optional
        Code(s) of the target channel(s).
    location: string or list, optional
        Code(s) of the target location(s).
    starttime: string or obspy.UTCDateTime, optional
        Target start time.
    endtime: string or obspy.UTCDateTime, optional
        Target end time.
    tag: string, default to 'raw'
        Tag name in the ASDF file.

    Returns
    -------
    traces: obspy.Stream
        The seismic data.
    """
    from obspy import Stream
    from pyasdf import ASDFDataSet

    traces = Stream()
    with ASDFDataSet(where, mode="r") as ds:
        for station_ in ds.ifilter(
            ds.q.tag == tag,
            ds.q.network == network,
            ds.q.station == station,
            ds.q.channel == f"*{channel}",
            ds.q.location == location,
        ):
            for tr in getattr(station_, tag):
                # traces += tr.slice(starttime=starttime,
                #        endtime=endtime, nearest_sample=True)
                net = tr.stats.network
                sta = tr.stats.station
                cha = tr.stats.channel
                loc = tr.stats.location
                traces += ds.get_waveforms(
                    network=net,
                    station=sta,
                    location=loc,
                    channel=cha,
                    starttime=starttime,
                    endtime=endtime,
                    tag=tag,
                )
    return traces


def data_reader_mseed(
    where,
    network="*",
    stations=["*"],
    channels=["*"],
    location="*",
    starttime=None,
    endtime=None,
    attach_response=False,
    data_folder="",
    data_files=None,
    channel_template_str="[A-Z][A-Z]",
    **kwargs,
):
    """Data reader for BPMF.

    This data reader is specifically designed for the folder tree convention
    that we use in the tutorial. We will use the same data reader at later
    stages of the workflow.

    Note: This is the data reader introduced in BPMF's tutorial.

    Parameters
    -----------
    where : str
        Path to data file or root data folder.
    network : str or list, optional
        Code(s) of the target network(s).
    stations : str or list, optional
        Code(s) of the target station(s).
    channels : str or list, optional
        Code(s) of the target channel(s).
    location : str or list, optional
        Code(s) of the target location(s).
    starttime : str or obspy.UTCDateTime, optional
        Target start time.
    endtime : str or obspy.UTCDateTime, optional
        Target end time.
    attach_response : bool, optional
        If True, find the instrument response from the xml files
        and attach it to the obspy.Stream output instance.
    data_folder : str, optional
        If given, is the child folder in `where` containing
        the mseed files to read.
    data_files : list, optional
        If not None, is the list of full paths (str) to the data files
        to read.
    channel_template_str : str, optional
        Data files are searched assuming the following naming convention:
        `full_channel_name = channel_template_str + channels[i]`
        By default, `channel_template_str='[A-Z][A-Z]'`, meaning that
        it is assumed that channel names start with two letters.

    Returns
    -------
    traces: obspy.Stream
        The seismic data.
    """
    import glob
    import os

    from obspy import Stream, read, read_inventory

    if not isinstance(stations, list) and not isinstance(stations, np.ndarray):
        stations = [stations]
    if not isinstance(channels, list) and not isinstance(channels, np.ndarray):
        channels = [channels]

    traces = Stream()
    # read your data into traces
    # data_files = []
    if data_files is None:
        data_files = []
        for sta in stations:
            for cha in channels:
                cha = channel_template_str + cha 
                data_files.extend(
                    glob.glob(
                        os.path.join(
                            where, data_folder, f"{network}.{sta}.{location}.{cha}[_.]*"
                        )
                    )
                )
    resp_files = set()
    for fname in data_files:
        #print(f"Reading from {fname}...")
        tr = read(fname, starttime=starttime, endtime=endtime, **kwargs)
        if len(tr) == 0:
            # typically because starttime-endtime falls into a gap
            continue
        traces += tr
        network, sta = tr[0].stats.network, tr[0].stats.station
        if attach_response:
            resp_files.update(
                set(glob.glob(os.path.join(where, "resp", f"{network}.{sta}.xml")))
            )
    if attach_response:
        invs = list(map(read_inventory, resp_files))
        traces.attach_response(invs)
    return traces
