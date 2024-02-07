import sys


def print_attr_sizes(obj, print_threshold=None):
    """
    Print sizes of attributes of an object that exceed a threshold
    :param obj: python object
    :param print_threshold: print sizes of attributes at or above this threshold
    """

    if print_threshold is None:
        print_threshold = 0

    print(f"The following object attributes exceed {print_threshold} MB:")
    for attr in dir(obj):
        attr_size = sys.getsizeof(getattr(obj, attr))/(10**6)  # mb
        if attr_size >= print_threshold:
            print(f"{attr} is {attr_size} MB")
