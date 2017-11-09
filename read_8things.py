import numpy as np

longform_y_data = []
with open('8things.pat', 'r') as fin:
    for line in fin.readlines():
        this_data = line.split()
        longform_y_data.append(map(int, this_data[this_data.index('|') + 1:]))

y_data = []
for i  in xrange(8):
    y_data.append([item for sublist in longform_y_data[i*4:(i+1)*4] for item in sublist])

y_data = np.array(y_data)
x_data = np.eye(len(y_data))

scrambles = [
[0, 1, 2, 3, 4, 5, 6, 7],
[3, 7, 1, 5, 2, 6, 0, 4],
[2, 3, 6, 7, 0, 1, 4, 5],
[4, 2, 7, 1, 6, 3, 5, 0]
]

scrambled_y_data = []
for i  in xrange(8):
    this_data = [longform_y_data[scrambles[j][i] * 4 + j] for j in xrange(4)]
    scrambled_y_data.append([item for sublist in this_data for item in sublist])
scrambled_y_data = np.array(scrambled_y_data)

if __name__ == "__main__":
    print longform_y_data
    print y_data
    print scrambled_y_data

