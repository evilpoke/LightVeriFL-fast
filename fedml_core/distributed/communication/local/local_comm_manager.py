"""Single-process ("SP") communication backend: the server and all clients live in ONE thread.

Every party owns a :class:`LocalCommunicationManager` endpoint of a shared :class:`LocalBus`.
``send_message`` puts the message into one FIFO queue (FIFO per sender/receiver pair, like MPI),
:meth:`LocalBus.run` delivers the messages one after another by calling the receiver's handler.
Messages are copied with pickle on the way (exactly what MPI does), so no party can accidentally
share objects with another one.

``on_send(params, sender, receiver)`` is an optional hook that is called for every message, e.g.
to measure the size of the transmitted objects (see fedml_api/distributed/utils/bench.py).
"""
import collections
import logging
import pickle

from ..base_com_manager import BaseCommunicationManager
from ..message import Message


class LocalBus(object):
    def __init__(self, on_send=None):
        self.on_send = on_send
        self.queue = collections.deque()
        self.endpoints = dict()
        self.stop_requested = False
        self.delivered = 0

    def endpoint(self, rank):
        ep = LocalCommunicationManager(self, rank)
        self.endpoints[rank] = ep
        return ep

    def post(self, msg: Message):
        params = msg.get_params()
        sender = params[Message.MSG_ARG_KEY_SENDER]
        receiver = params[Message.MSG_ARG_KEY_RECEIVER]
        if self.on_send is not None:
            self.on_send(params, sender, receiver)
        copied = Message()
        copied.init(pickle.loads(pickle.dumps(params, protocol=pickle.HIGHEST_PROTOCOL)))
        self.queue.append((receiver, copied))

    def run(self):
        """Deliver messages until the queue is empty (exceptions of a handler propagate)."""
        while self.queue:
            receiver, msg = self.queue.popleft()
            self.endpoints[receiver].notify(msg)
            self.delivered += 1
        logging.info("LocalBus: %d messages delivered (finish requested: %s)" % (self.delivered, self.stop_requested))

    def stop(self):
        # graceful: messages that are already queued (e.g. the server's last broadcast) are still delivered
        self.stop_requested = True


class LocalCommunicationManager(BaseCommunicationManager):
    def __init__(self, bus: LocalBus, rank):
        self.bus = bus
        self.rank = rank
        self._observers = []

    def send_message(self, msg: Message):
        self.bus.post(msg)

    def add_observer(self, observer):
        self._observers.append(observer)

    def remove_observer(self, observer):
        self._observers.remove(observer)

    def handle_receive_message(self):
        # nothing to do: LocalBus.run() drives the delivery for all parties
        pass

    def stop_receive_message(self):
        self.bus.stop()

    def notify(self, msg: Message):
        msg_type = msg.get_type()
        for observer in self._observers:
            observer.receive_message(msg_type, msg)
