import os

import numpy as np

for file in sorted(os.listdir(".")):
    if file.endswith(".txt"):
        with open(file, "r") as f:
            values = list(map(lambda x: float(x[-6:]),  f.readlines()))
        print("{:20s}: mean={:.1f}, std={:.1f}".format(
            file[:-4], np.mean(values)*100, np.std(values)*100))
