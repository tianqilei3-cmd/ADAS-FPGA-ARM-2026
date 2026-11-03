"""Linux exclusive serial I/O with V3 identity/action confirmation."""
import os
import select
import termios
import time

from uart_confirmed import ConfirmedUart as LegacyConfirmedUart
from uart_v2 import Uart as RawUart
from basic70_wire import ReplyStream,decode_feedback,feedback_matches,query_wire
from transport_age_guard import refresh_context


class ConfirmedUart(LegacyConfirmedUart):
    def __init__(self,path,ack_timeout=.05):
        super().__init__(path,ack_timeout=ack_timeout)
        self.latest_feedback=None
        try:
            feedback,transaction=self.query_state()
            if feedback['version']!=3 or feedback['build_tag'] not in (0x28,0x29) or feedback['capabilities']&31!=31:
                raise OSError('FPGA firmware does not support Basic70 V3')
            self.latest_feedback=feedback
        except BaseException:
            self.close();raise

    def _transaction(self,data,expected,fresh_until=None):
        before=b''
        if select.select([self.fd],[],[],0)[0]:before=os.read(self.fd,4096)
        started=time.monotonic()
        result=dict(command=data[1],sent=False,acknowledged=False,tx_started_monotonic=None,
            ack_observed_monotonic=None,reply_hex='',reply_reads=[],reply_commands=[],
            unexpected_before_tx_hex=before.hex(' '),invalid_reply_bytes=0,
            partial_reply_hex='',reason='stale_before_tx')
        if fresh_until is not None and started>=fresh_until:return result,[]
        result['tx_started_monotonic']=started
        written=0
        while written<len(data):
            try:count=os.write(self.fd,data[written:])
            except BlockingIOError:
                if not select.select([],[self.fd],[],self.ack_timeout)[1]:raise OSError('UART write timeout')
                continue
            if count<=0:raise OSError('UART write returned no progress')
            written+=count
        termios.tcdrain(self.fd)
        initial=b'';result['sent']=True
        # The complete checked ACK establishes receiver readiness. No fixed sleep.
        parser=ReplyStream();frames=[];raw=bytearray();deadline=time.monotonic()+self.ack_timeout
        def consume(chunk):
            observed=time.monotonic();raw.extend(chunk);new=parser.feed(chunk);frames.extend(new)
            result['reply_reads'].append(dict(observed_monotonic=observed,hex=chunk.hex(' ')))
            result['reply_commands'].extend(frame[1] for frame in new)
            if any(frame[1]==expected for frame in new):result['ack_observed_monotonic']=observed
        if initial:consume(initial)
        while result['ack_observed_monotonic'] is None and not any(0xe0<=x<=0xef for x in result['reply_commands']):
            remaining=deadline-time.monotonic()
            if remaining<=0 or not select.select([self.fd],[],[],remaining)[0]:break
            chunk=os.read(self.fd,4096)
            if not chunk:raise OSError('UART EOF')
            consume(chunk)
        has_error=any(0xe0<=x<=0xef for x in result['reply_commands'])
        result.update(acknowledged=result['ack_observed_monotonic'] is not None and not has_error,
            reply_hex=bytes(raw).hex(' '),invalid_reply_bytes=parser.invalid_bytes,
            partial_reply_hex=bytes(parser.buffer).hex(' '),exchange_ms=(time.monotonic()-started)*1000)
        result['reason']='confirmed' if result['acknowledged'] else 'fpga_protocol_error' if has_error else 'ack_timeout'
        return result,frames

    def query_keys(self):
        from physical_keys import decode_keys
        transaction,frames=self._transaction(bytes.fromhex('55 03 58'),3)
        if not transaction['acknowledged']:raise OSError('FPGA key query failed: '+transaction['reply_hex'])
        snapshot=decode_keys(next(f for f in frames if f[1]==3))
        snapshot['query']=transaction
        return snapshot

    def query_state(self):
        transaction,frames=self._transaction(query_wire(),0x84)
        if not transaction['acknowledged']:raise OSError('Identified FPGA feedback query failed: '+transaction['reply_hex'])
        feedback=decode_feedback(next(frame for frame in frames if frame[1]==0x84))
        return feedback,transaction

    def exchange(self,data,fresh_until=None):
        if len(data)<4 or data[0]!=0x55 or sum(data[:-1])&255!=data[-1]:
            raise ValueError('Complete checked wire packet required')
        data,age_guard=refresh_context(data,fresh_until,time.monotonic())
        transaction,frames=self._transaction(data,data[1],fresh_until=fresh_until)
        if age_guard:transaction.update(transport_age_guard=age_guard,wire_hex=data.hex(' '))
        if data[1]!=0x13 or not transaction['acknowledged']:return transaction
        try:
            feedback,query=self.query_state()
        except (OSError,ValueError) as error:
            transaction.update(acknowledged=False,reason='feedback_query_failed',feedback_error=repr(error))
            return transaction
        transaction['reply_hex']+=' '+query['reply_hex']
        transaction['reply_reads'].extend(query['reply_reads'])
        transaction['reply_commands'].extend(query['reply_commands'])
        transaction['invalid_reply_bytes']+=query['invalid_reply_bytes']
        transaction['feedback']=feedback
        transaction['feedback_query_unexpected_before_tx_hex']=query['unexpected_before_tx_hex']
        matched=feedback_matches(feedback,data)
        timely=fresh_until is None or time.monotonic()<fresh_until
        transaction.update(acknowledged=matched and timely,
            ack_observed_monotonic=query['ack_observed_monotonic'] if matched and timely else None,
            reason='confirmed' if matched and timely else 'feedback_expired' if not timely else 'feedback_rejected_or_identity_mismatch')
        transaction['exchange_ms']=(time.monotonic()-transaction['tx_started_monotonic'])*1000
        self.latest_feedback=feedback
        return transaction
