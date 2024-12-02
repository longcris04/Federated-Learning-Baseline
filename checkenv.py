# import torch
# print(torch.cuda.is_available())  # Should print True if CUDA is properly installed
# print(torch.cuda.get_device_name(0))  # Should print the name of your GPU, e.g., 'NVIDIA RTX 3060'

import torch
import numpy as np
import random
import matplotlib.pyplot as plt

# print("PyTorch version:", torch.__version__)
# print("CUDA available:", torch.cuda.is_available())
# generator1 = torch.Generator().manual_seed(42)
# generator2 = torch.Generator().manual_seed(42)
# random_split(range(10), [3, 7], generator=generator1)
# random_split(range(30), [0.3, 0.3, 0.4], generator=generator2)
# random.seed(10)
# arr = np.random.rand(10)
# #print(arr)
# data = torch.utils.data.random_split(arr, [0.4,0.6])
# for subset in data:
#     print(subset.dataset[subset.indices])

# Create some data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# Create the first figure and plot y1
plt.figure(1)  # Figure 1
plt.plot(x, y1, 'r-')  # red line
plt.title('Sine Wave')
plt.xlabel('X')
plt.ylabel('sin(X)')

# Create the second figure and plot y2
plt.figure(2)  # Figure 2
plt.plot(x, y2, 'b-')  # blue line
plt.title('Cosine Wave')
plt.xlabel('X')
plt.ylabel('cos(X)')

# Show all figures
plt.show()