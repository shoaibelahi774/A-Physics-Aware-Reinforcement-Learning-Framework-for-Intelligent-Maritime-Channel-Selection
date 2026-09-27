from channels.channel import Channel

class WiFi(Channel):
    def __init__(self) -> None:
        super().__init__(
            name="WiFi",
            latency=10.0,
            bandwidth=200.0,
            cost=0.50,
            stability=0.50,
        )
