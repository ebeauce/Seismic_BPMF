def data_reader_template(
    where,
    network="*",
    station="*",
    channel="*",
    location="*",
    starttime="*",
    endtime="*",
    **kwargs
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
    **kwargs
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
            ds.q.channel == channel,
            ds.q.location == location,
        ):
            for tr in getattr(station_, tag):
                #traces += tr.slice(starttime=starttime,
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

