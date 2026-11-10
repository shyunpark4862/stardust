"""The missing value (API.md §2): one singleton, ``sdpd.NA``, for every dtype."""


class NAType:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self):
        return "<NA>"

    def __bool__(self):
        raise TypeError("the truth value of NA is ambiguous")

    def __reduce__(self):  # pickling keeps the singleton
        return "NA"


NA = NAType()
