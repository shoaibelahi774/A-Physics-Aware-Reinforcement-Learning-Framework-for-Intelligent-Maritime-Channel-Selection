from channels.channel import Channel

class Cellular4G(Channel):
    def __init__(self) -> None:
        super().__init__(
            name="4G Cellular",
            latency=40.0,
            bandwidth=100.0,
            cost=1.0,
            stability=0.70,
        )
