"""Message transport abstraction: in-process queue (default, used in tests/demo) or Kafka/Redpanda (requires a running broker)."""
from __future__ import annotations
import json, queue


class InMemoryTransport:
    def __init__(self): self.topics: dict[str, queue.Queue] = {}

    def _q(self, t): return self.topics.setdefault(t, queue.Queue())
    def send(self, topic, msg: dict): self._q(topic).put(json.dumps(msg))

    def poll(self, topic, timeout=0.01):
        try: return json.loads(self._q(topic).get(timeout=timeout))
        except queue.Empty: return None

    def close(self): pass


class KafkaTransport:
    """kafka-python wrapper. Start a broker with `docker compose up redpanda` (see docker/docker-compose.yml). Not exercised in CI."""

    def __init__(self, bootstrap: str, group: str = "presage"):
        from kafka import KafkaProducer, KafkaConsumer
        self.bootstrap, self.group = bootstrap, group
        self.producer = KafkaProducer(bootstrap_servers=bootstrap, value_serializer=lambda v: json.dumps(v).encode())
        self.consumers = {}

    def send(self, topic, msg): self.producer.send(topic, msg)

    def poll(self, topic, timeout=0.5):
        from kafka import KafkaConsumer
        if topic not in self.consumers:
            self.consumers[topic] = KafkaConsumer(topic, bootstrap_servers=self.bootstrap, group_id=self.group, auto_offset_reset="earliest",
                                                  value_deserializer=lambda b: json.loads(b.decode()), consumer_timeout_ms=int(timeout * 1000))
        for m in self.consumers[topic]: return m.value
        return None

    def close(self): self.producer.flush()
