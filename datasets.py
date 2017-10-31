from __future__ import print_function
from __future__ import absolute_import
from __future__ import division 
import numpy

def random_dataset(num_examples, num_outputs):
    """Returns num_examples random unit length vectors of length num_outputs"""
    unscaled = 2*numpy.random.rand(num_examples, num_outputs) - 1 
    scaled = unscaled / numpy.expand_dims(numpy.sqrt(numpy.sum(unscaled**2, axis=1)), axis=1)
    return scaled


def correlated_dataset(num_examples, num_outputs, num_domains, q):
    """Returns a dataset of num_domains block domains which are correlated at strength q"""
    shared_structure_set = random_dataset(num_examples, num_outputs)
    random_structure_sets = [random_dataset(num_examples, num_outputs) for _ in xrange(num_domains)]

    if 0 <= q <= 1:
	shared_weight = numpy.sqrt(q)
	unshared_weight = numpy.sqrt(1-q)
	domain_sets = [shared_structure_set * shared_weight + random_structure_set * unshared_weight for random_structure_set in random_structure_sets]
    elif -1 <= q < 0:
	q = -q
	shared_weight = numpy.sqrt(q)
	unshared_weight = numpy.sqrt(1-q)
	domain_sets = [(-1 if i % 2 else 1) * shared_structure_set * shared_weight +  random_structure_set * unshared_weight for i, random_structure_set in enumerate(random_structure_sets)]
    else:
	raise ValueError("Error, correlation should be between -1 and 1")

    y_data = numpy.concatenate(domain_sets, axis=1)
    x_data = numpy.eye(len(y_data))
    return x_data, y_data


def transformed_dataset(num_examples, num_outputs, num_domains):
    """Returns a dataset of num_domains block domains which are all transformations of the first -- not orthogonal transformations yet"""
    shared_structure_set = random_dataset(num_examples, num_outputs)
    transforms = [random_dataset(num_examples, num_examples) for _ in xrange(num_domains - 1)]

    domain_sets = [shared_structure_set] + [numpy.dot(transform, shared_structure_set) for transform in transforms]

    y_data = numpy.concatenate(domain_sets, axis=1)
    x_data = numpy.eye(len(y_data))
    return x_data, y_data

if __name__ == "__main__":
    print(correlated_dataset(2, 3, 2, 1))
    print(correlated_dataset(2, 3, 2, 0.5))
    print(correlated_dataset(2, 3, 2, -1))

    print()
    print(transformed_dataset(2, 3, 2))

