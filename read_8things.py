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

if __name__ == "__main__":
    print longform_y_data
    print y_data

