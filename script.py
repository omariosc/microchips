# Run cla.py with the following command:
# python cla.py --input_folder input --output_folder output
# With inputs in folder = ["0", "1", "2", "A", "B", "C", "D", "E", "X"]

import os

for i in ["0", "1", "2", "A", "B", "C", "D", "E", "X"]:
    os.system(f"python3.12 cla.py --folder {i}")
    print(f"Done for {i}")
