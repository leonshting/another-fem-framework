class VerbosePrinter:
    def __init__(self, **kwargs):
        self._verbose = kwargs.get('verbose', False)

    def _verbose_print(self, msg):
        if(self._verbose):
            print(msg)

