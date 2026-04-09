from dt4acc_lib.model.output.tune import Tune


class TuneConversion:
    """Convert from Floquet coordinates to frequency

    Assuming that x and y use the same scaling. This is typically
    the case as the calculation needs:

    * (ring) revolution frequency

    This is typically calculated from

    * main RF / master clock frequency
    * number of buckets

    Thus, it should be the same for both planes

    Warning:
        Need to review connection to master clock
    """
    def __init__(self, to):
        self.to = to

    def forward(self, inp: Tune) ->  Tune:
        return Tune(x=self.to.forward(inp.x), y=self.to.forward(inp.y))

    def inverse(self, inp: Tune) ->  Tune:
        return Tune(x=self.to.forward(inp.x), y=self.to.forward(inp.y))
