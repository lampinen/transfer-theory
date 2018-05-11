from __future__ import print_function
from __future__ import absolute_import
from __future__ import division 
import numpy
from scipy.linalg import block_diag
from orthogonal_matrices import random_orthogonal

def random_dataset(num_examples, num_outputs):
    """Returns num_examples random unit length vectors of length num_outputs"""
    result = numpy.random.randn(num_examples, num_outputs)/num_outputs
    return result


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
    transforms = [random_orthogonal(num_examples) for _ in xrange(num_domains - 1)]

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


def random_vector_at_cos(initial_vec, cos_theta):
    """Calculates a uniformly distributed unit vector at a given cosine of angle to the initial vector"""
    initial_vec /= numpy.linalg.norm(initial_vec)
    random_orthogonal_vec = numpy.random.randn(len(initial_vec))
    random_orthogonal_vec /= numpy.linalg.norm(random_orthogonal_vec)
    random_orthogonal_vec -= numpy.dot(initial_vec, random_orthogonal_vec) * initial_vec 
    random_orthogonal_vec /= numpy.linalg.norm(random_orthogonal_vec)
    new_vec = cos_theta * initial_vec + (1 - cos_theta) * random_orthogonal_vec
    return new_vec

def random_rotation_matrix(theta, n):
    """Generates a 'rotation' of an angle theta in n dimensions about many random axes"""
    P = random_orthogonal(n)
    if n % 2 == 0:
        R = block_diag(*([numpy.array([[numpy.cos(theta), -numpy.sin(theta)],[numpy.sin(theta), numpy.cos(theta)]])]*(n//2))) 
    else: 
        R = block_diag(*([numpy.array([[numpy.cos(theta), -numpy.sin(theta)],[numpy.sin(theta), numpy.cos(theta)]])]*(n//2) + [numpy.eye(1)])) 
    return numpy.matmul(numpy.transpose(P),  numpy.matmul(R, P))


def shared_input_modes_dataset(num_examples, num_outputs, num_domains, q, num_nonempty=4):
    """Returns a dataset of num_domains block domains with input modes related at angle q"""
    if num_outputs < num_examples:
        raise ValueError("Less than full rank tasks are not supported at the moment (the appropriate truncation of the permuted S matrix is kinda annoying)")
    input_modes = random_orthogonal(num_examples)
    strengths = numpy.array(range(num_nonempty, 0, -1) + [0.]* (num_examples - num_nonempty))
    
    def _strengths_to_S(strengths, num_outputs=num_outputs):
        return numpy.concatenate((numpy.diag(strengths), numpy.zeros((num_outputs-len(strengths), num_examples))), axis=0)

    S = _strengths_to_S(strengths)

    rotations = [numpy.eye(num_examples)] + [random_rotation_matrix(q, num_examples) for i in xrange(num_domains - 1)]
    domain_sets = [numpy.transpose(numpy.matmul(random_orthogonal(num_outputs), numpy.matmul(S, numpy.matmul(this_rotation, input_modes)))) for this_rotation in rotations]

    y_data = numpy.concatenate(domain_sets, axis=1)
    x_data = numpy.eye(len(y_data))
    return x_data, y_data, input_modes

def noisy_shared_input_modes_dataset(num_examples, num_outputs, num_domains, q, num_nonempty=4, noise_var=0.1):
    x_data, y_data, input_modes = shared_input_modes_dataset(num_examples, num_outputs, num_domains, q, num_nonempty=num_nonempty)
    y_data_noisy = y_data + numpy.sqrt(noise_var) * numpy.random.standard_normal(numpy.shape(y_data))
    return x_data, y_data, y_data_noisy, input_modes

def noisy_shared_input_modes_dataset_different_inputs(num_examples, num_outputs, num_domains, q, num_nonempty=4, noise_var=0.1, input_type="orthogonal"):
    _, y_data, y_data_noisy, input_modes = noisy_shared_input_modes_dataset(num_examples, num_outputs, num_domains, q, num_nonempty=num_nonempty, noise_var=noise_var)
    if input_type == "orthogonal":
        x_data = random_orthogonal(len(y_data)) 
    elif input_type == "gaussian":
        x_data = np.random.randn(num_examples, num_examples) 
    else:
        raise ValueError("Unknown input type!")
    return x_data, y_data, y_data_noisy, input_modes

def SVD_dataset(num_examples, num_outputs, num_nonempty=4, singular_value_multiplier=1):
    """Like the shared input modes dataset, but only one domain"""
    input_modes = random_orthogonal(num_examples)
    strengths = singular_value_multiplier * numpy.array(range(num_nonempty, 0, -1) + [0.]* (num_examples - num_nonempty))
    
    def _strengths_to_S(strengths, num_outputs=num_outputs):
        if num_outputs > num_examples:
            return numpy.concatenate((numpy.diag(strengths), numpy.zeros((num_outputs-len(strengths), num_examples))), axis=0)
        else:
            return numpy.diag(strengths)[:num_outputs, :]

    S = _strengths_to_S(strengths)

    y_data = numpy.transpose(numpy.matmul(random_orthogonal(num_outputs), numpy.matmul(S, input_modes)))
    x_data = numpy.eye(len(y_data))
    return x_data, y_data, input_modes

def noisy_SVD_dataset(num_examples, num_outputs, num_nonempty=4, noise_var=0.1, singular_value_multiplier=1):
    x_data, y_data, input_modes = SVD_dataset(num_examples, num_outputs, num_nonempty=num_nonempty, singular_value_multiplier=singular_value_multiplier)
    y_data_noisy = y_data + numpy.sqrt(noise_var) * numpy.random.standard_normal(numpy.shape(y_data))
    _, S, _ = numpy.linalg.svd(y_data)
    print(numpy.amax(S))
    return x_data, y_data, y_data_noisy, input_modes

def noisy_SVD_dataset_different_inputs(num_examples, num_outputs, num_nonempty=4, noise_var=0.1, singular_value_multiplier=1, input_type="orthogonal"):
    x_data, y_data, y_data_noisy, input_modes = noisy_SVD_dataset(num_examples, num_outputs, num_nonempty=num_nonempty, noise_var=noise_var, singular_value_multiplier=singular_value_multiplier)
    if input_type == "orthogonal":
        x_data = random_orthogonal(len(y_data)) 
    elif input_type == "gaussian":
        x_data = np.random.randn(num_examples, num_examples) 
    else:
        raise ValueError("Unknown input type!")
    return x_data, y_data, y_data_noisy, input_modes
