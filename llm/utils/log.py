class LOG():
    @classmethod
    def log(cls, *args, head=True):
        if head: print("\n*************************************************************")
        for item in args:
            print(item)

    @classmethod
    def append(cls, *args):
        LOG.log(*args, head=False)