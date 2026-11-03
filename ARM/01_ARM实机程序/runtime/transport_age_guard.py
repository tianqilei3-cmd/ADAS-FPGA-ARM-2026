"""Conservative V3 age at the transport boundary; never renew image time."""
import math


def refresh_context(data,fresh_until,now,transport_margin=.010,grant_reserve=.050):
    if data[:3]!=bytes([0x55,0x13,16]) or len(data)!=20:return data,None
    if fresh_until is None:return data,None
    elapsed=max(0.,now-(fresh_until-.5));wire=bytearray(data)
    # One age unit is 5ms. The lease starts at the receiver after serialization.
    wire[17]=min(255,max(wire[17],math.ceil((elapsed+transport_margin)/.005)))
    suppressed=wire[18]==2 and elapsed>.5-grant_reserve
    if suppressed:wire[18]=1;wire[8]&=~64
    wire[-1]=sum(wire[:-1])&255
    return bytes(wire),dict(image_age_before_write_ms=elapsed*1000,transport_margin_ms=transport_margin*1000,
        encoded_age_units=wire[17],grant_suppressed_for_transport_budget=suppressed)
