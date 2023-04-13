import numpy

def entropy(image: numpy.ndarray) -> numpy.ndarray:

    _, counts = numpy.unique(image, return_counts=True)
    total = len(counts)
    frequencies = counts / total
    entropy = (-frequencies * numpy.log2(frequencies)).sum()

    return entropy