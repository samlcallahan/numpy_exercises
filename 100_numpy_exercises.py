import numpy as np

print(np.__version__)
np.show_config()

Z = np.zeros(10)
print(Z)

memory = Z.size * Z.itemsize

np.info(np.add)

Z[4] = 1

vector = np.arange(10,50)

reverse = np.flip(vector)

matrix = np.arange(9).reshape(3,3)

non_zero = np.nonzero([1,2,0,0,4,0])

identity = np.eye(3)

three_D_array = np.random.rand(3,3,3)

random_array = np.random.rand(10,10)
random_array.min()
random_array.max()

thirty = np.arange(30)
thirty.mean()

zeros_in_ones = np.ones((10,10))
zeros_in_ones[1:-1,1:-1] = 0

np.pad(zeros_in_ones,1,'constant')

0 * np.nan # nan
np.nan == np.nan # false
np.inf > np.nan # false
np.nan - np.nan # nan
np.nan in set([np.nan]) # true
0.3 == 3 * 0.1 # false

diagonal = np.diag(np.arange(5),-1)

checkerboard = np.zeros((8,8))
checkerboard[::2,::2] = 1
checkerboard[1::2,1::2] = 1

hundredth_index = np.unravel_index(99,(6,7,8))

one_tile = np.array([[0,1],[1,0]])
tiled = np.tile(one_tile,(4,4))

not_yet_normal = np.random.rand(5,5)
normal = (not_yet_normal - not_yet_normal.mean()) / not_yet_normal.std()

colors = np.dtype([('r',np.ubyte,1),
                   ('g',np.ubyte,1),
                   ('b',np.ubyte,1),
                   ('a',np.ubyte,1)])

five_by_three = np.random.rand(5,3)
three_by_two = np.random.rand(3,2)
dot_product = five_by_three @ three_by_two

one_d_array = np.arange(20)
one_d_array[(one_d_array > 3) & (one_d_array <= 8)] *= -1

print(sum(range(5),-1)) # prints 9 (summing range and -1)
from numpy import *
print(sum(range(5),-1)) # prints 10 (sum iterating over last column of the range)

Z**Z # returns array of each element raised to the power of itself
2 << Z >> 2 # returns array of (2 shifted z digits left, bitwise) shifted 2 digits right, bitwise
Z <- Z # returns array of whether each element is nonnegative
1j*Z # returns array of each element multiplied by i
Z/1/1 # returns the original array
Z<Z>Z # invalid

np.array(0) / np.array(0) # returns nan because 0 divided by 0 is undefined
np.array(0) // np.array(0) # returns 0
np.array([np.nan]).astype(int).astype(float) # returns array([maximum negative number])

to_round = np.random.uniform(-10,10,(10,10))
rounded = np.copysign(np.ceil(np.abs(to_round)),to_round)

set_A = np.random.randint(-10,10,(10,10))
set_B = np.random.randint(-10,10,(10,10))
common_values = np.intersect1d(set_A,set_B)

defaults = np.seterr(all="ignore")
worst_settings_possible = np.seterr(**defaults)

np.sqrt(-1) == np.emath.sqrt(-1) # false

today = np.datetime64('today')
yesterday = today - 1
tomorrow = today + 1

dates_in_july_2016 = np.arange(np.datetime64('2016-07','D'),np.datetime64('2016-08'))

A = np.random.rand(3,3)
B = np.random.rand(3,3)
np.add(A,B,out=B)
np.divide(A,-2,out=A)
np.multiply(A,B,out=A)

array = np.random.uniform(0,10,10)
print(array - array%1)
print(array.astype(int))
print(np.floor(array))
print(np.ceil(array)-1)
print(np.trunc(array))

matrix = np.zeros((5,5))
matrix += np.arange(5)

def generate():
    for x in range(10):
        yield x

array_10 = np.fromiter(generate(),int)

vector_0_to_1 = np.linspace(0,1,11,endpoint=False)[1:]

sorted_array = np.sort(np.random.rand(10))

sum_array = np.add.reduce(np.arange(10))

array_A = np.random.randint(3,45,10)
array_B = np.random.randint(3,45,10)
equal = np.array_equal(array_A,array_B)

array_A.flags.writeable = False

cartesian = np.random.randint(0,10,(10,2))
radii = np.sqrt(cartesian[:,0]**2 + cartesian[:,1]**2)
angles = np.arctan2(cartesian[:,1],cartesian[:,0])
polar = np.vstack((radii,angles)).T

random_vector = np.random.rand(10)
random_vector[random_vector.argmax()] = 0

