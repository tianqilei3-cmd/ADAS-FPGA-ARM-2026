"""FPGA reply parsing. Replies identify commands, not frame IDs or bbox slots."""


class ReplyParser:
    def __init__(self):
        self.buffer = bytearray()
        self.invalid_bytes = 0

    def feed(self, data):
        self.buffer.extend(data)
        commands = []
        while self.buffer:
            if self.buffer[0] != 0x55:
                del self.buffer[0]
                self.invalid_bytes += 1
                continue
            if len(self.buffer) < 2:
                break
            # KEY_CMD has a five-byte response; it must not be mistaken for an ACK.
            size = 5 if self.buffer[1] == 0x03 else 3
            if len(self.buffer) < size:
                break
            reply = self.buffer[:size]
            if sum(reply[:-1]) & 0xff != reply[-1]:
                del self.buffer[0]
                self.invalid_bytes += 1
                continue
            commands.append(reply[1])
            del self.buffer[:size]
        return commands


def exchange_frame(uart, wires, decoded_at):
    """One outstanding packet at a time. Stop the batch at the first failure.

    No automatic retransmission: it could send an old visual decision. A missing
    status ACK prevents bbox transmission. The caller closes and later reopens
    a failed link before attempting a newly inferred frame.
    """
    result = dict(status_packet_sent=False, status_tx_at=None,
                  status_acknowledged=False, status_ack_observed_at=None,
                  boxes_expected=max(0, len(wires) - 1), boxes_attempted=0,
                  boxes_acknowledged=0, packets_written=0,
                  all_acknowledged=False, transactions=[], reply_hex='',
                  failure_reason=None, reopen_required=False,
                  link_state='stale_not_sent')
    replies = []
    for index, wire in enumerate(wires):
        try:
            transaction = uart.exchange(wire, fresh_until=decoded_at + .5)
        except OSError as error:
            # An OS write/drain exception may occur after some bytes were sent.
            # Do not count it as a complete written packet or a confirmed frame.
            result.update(failure_reason='io_error', io_error=repr(error),
                          reopen_required=True, link_state='io_error')
            break
        result['transactions'].append(transaction)
        if transaction['sent']:
            result['packets_written'] += 1
            if index == 0:
                result['status_packet_sent'] = True
                result['status_tx_at'] = transaction['tx_started_monotonic']
            else:
                result['boxes_attempted'] += 1
        if transaction['reply_hex']:
            replies.append(transaction['reply_hex'])
        if index == 0 and transaction['acknowledged']:
            result['status_acknowledged'] = True
            result['status_ack_observed_at'] = transaction['ack_observed_monotonic']
        if index > 0 and transaction['acknowledged']:
            result['boxes_acknowledged'] += 1
        if not transaction['acknowledged']:
            result['failure_reason'] = transaction['reason']
            result['reopen_required'] = transaction['sent']
            result['link_state'] = ('status_ok_boxes_unconfirmed' if result['status_acknowledged']
                                    else 'unconfirmed' if transaction['sent'] else 'stale_not_sent')
            break
    else:
        result.update(all_acknowledged=True, link_state='confirmed')
    result['reply_hex'] = ' '.join(replies)
    return result


def link_text(receipt, persons, port_available):
    if receipt is None:
        return 'UART: waiting for port; person flag=' + str(int(persons > 0))
    state = receipt['link_state'].upper()
    return (f'UART {state} | status ACK={int(receipt["status_acknowledged"])} '
            f'boxes={receipt["boxes_acknowledged"]}/{receipt["boxes_expected"]} '
            f'| person flag={int(persons > 0)}')
