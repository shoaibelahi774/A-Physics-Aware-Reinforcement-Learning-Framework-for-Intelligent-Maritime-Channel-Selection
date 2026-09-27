from channels.channel import Channel

class ShortRangeRadio(Channel):
    def __init__(self) -> None:
        super().__init__(
            name="Short-Range Radio",
            latency=5.0,
            bandwidth=5.0,
            cost=0.10,
            stability=0.40,
        )
