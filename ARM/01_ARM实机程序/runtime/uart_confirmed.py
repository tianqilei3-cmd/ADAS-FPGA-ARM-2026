"""ACK-aware wrapper; the original V2 encoder and 5 ms send gap are retained.

    Linux exclusive mode: https://man7.org/linux/man-pages/man2/TIOCEXCL.2const.html
    CAP_SYS_ADMIN can bypass exclusivity; this does not prove the absence of an
    existing reader. The launcher additionally checks current /proc fd owners.
"""
import fcntl
import os
import select
import termios
import time

from ack_protocol import ReplyParser
from uart_v2 import Uart as OriginalUart


class ConfirmedUart(OriginalUart):
    def __init__(self, path, ack_timeout=.03):
        if not .005 <= ack_timeout <= .2:
            raise ValueError('ACK timeout must be between 5 and 200 ms')
        self.fd = None
        self.exclusive = False
        self.ack_timeout = ack_timeout
        try:
            super().__init__(path)
            fcntl.ioctl(self.fd, termios.TIOCEXCL)
            self.exclusive = True
        except BaseException:
            self.close()
            raise

    def close(self):
        if self.fd is not None:
            try:
                if self.exclusive:
                    try:
                        fcntl.ioctl(self.fd, termios.TIOCNXCL)
                    except OSError:
                        pass  # A physically disconnected device can reject ioctls.
            finally:
                os.close(self.fd)
                self.fd = None
                self.exclusive = False

    def exchange(self, data, fresh_until=None):
        if len(data) < 4 or data[0] != 0x55:
            raise ValueError('Expected a complete V2 wire packet')
        # Stale/unsolicited bytes already queued before this write get no credit.
        before = b''
        if select.select([self.fd], [], [], 0)[0]:
            before = os.read(self.fd, 4096)
        started = time.monotonic()
        result = dict(command=data[1], sent=False, acknowledged=False,
                      tx_started_monotonic=None, ack_observed_monotonic=None,
                      reply_hex='', reply_reads=[], reply_commands=[],
                      unexpected_before_tx_hex=before.hex(' '), invalid_reply_bytes=0,
                      partial_reply_hex='', reason='stale_before_tx')
        if fresh_until is not None and started >= fresh_until:
            return result
        result['tx_started_monotonic'] = started
        # Original send includes write, drain, the original 5 ms gap, and an
        # immediate read. The timeout below bounds additional ACK waiting;
        # it is not a hard real-time bound on kernel write/drain operations.
        initial = super().send(data)
        result['sent'] = True
        parser = ReplyParser()
        collected = bytearray()
        deadline = time.monotonic() + self.ack_timeout

        def consume(chunk):
            observed = time.monotonic()
            collected.extend(chunk)
            commands = parser.feed(chunk)
            result['reply_commands'].extend(commands)
            result['reply_reads'].append(dict(observed_monotonic=observed, hex=chunk.hex(' ')))
            if data[1] in commands and result['ack_observed_monotonic'] is None:
                result['ack_observed_monotonic'] = observed

        if initial:
            consume(initial)
        while result['ack_observed_monotonic'] is None and not any(
                0xe0 <= command <= 0xe6 for command in result['reply_commands']):
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.fd], [], [], remaining)[0]:
                break
            chunk = os.read(self.fd, 4096)
            if not chunk:
                raise OSError('UART returned EOF while awaiting ACK')
            consume(chunk)
        has_error = any(0xe0 <= command <= 0xe6 for command in result['reply_commands'])
        result.update(reply_hex=bytes(collected).hex(' '), invalid_reply_bytes=parser.invalid_bytes,
                      partial_reply_hex=bytes(parser.buffer).hex(' '),
                      acknowledged=result['ack_observed_monotonic'] is not None and not has_error)
        result['reason'] = ('confirmed' if result['acknowledged'] else
                            'fpga_protocol_error' if has_error else 'ack_timeout')
        result['exchange_ms'] = (time.monotonic() - started) * 1000
        return result
