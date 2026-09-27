import random
import time


PRIORITY_WEIGHTS = {
    "critical": 3,
    "normal": 2,
    "background": 1,
}


class DataPacket:
    def __init__(self, packet_type, size):
        self.packet_type = packet_type
        self.size = size
        self.timestamp = time.time()
        self.priority = PRIORITY_WEIGHTS[packet_type]

    @staticmethod
    def generate_random_packet():
        packet_types = ["critical", "normal", "background"]
        packet_type = random.choices(packet_types, weights=[0.2, 0.5, 0.3])[0]
        size = random.uniform(0.5, 5)
        return DataPacket(packet_type, size)