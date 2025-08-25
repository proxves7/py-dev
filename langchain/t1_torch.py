from __future__ import print_function
import torch
x = torch.rand(5, 3)
print(x)
y = torch.rand(2, 4)
print(y)
z = torch.rand(5, 3)
print(z)
t = x + z
print(t)

q1 = [0, 1, 2, 3]
q2 = (0, 1, 2, 3)
q3 = {"name": 1, "age": 2}
print(q1)
print(q2)
print(q3)