import struct
filename = "0010000M.track.eep"

data_lines = []
with open(filename, 'r') as f:
    for line in f:
        line = line.strip()
        if not line or line.startswith('#'):
            continue  # skip empty or comment lines
        data_lines.append(line)

data = []
for line in data_lines:
    # Split by whitespace, convert each value to float
    row = [float(x) for x in line.split()]
    data.append(row)

import numpy as np
import pandas as pd

data_array = np.array(data)
print(data_array.shape)  # e.g. (number_of_rows, number_of_columns)
df = pd.DataFrame(data_array)
df.head()