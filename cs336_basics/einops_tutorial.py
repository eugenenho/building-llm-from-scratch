# Examples are given for numpy. This code also setups ipython/jupyter/jupyterlite
# so that numpy arrays in the output are displayed as images
import numpy as np
from einops import rearrange, reduce, einsum
import torch 

x = np.random.RandomState(42).normal(size=[10, 32, 100, 200])
x = torch.from_numpy(x)
x.requires_grad = True

# print(type(x), x.shape)

y = rearrange(x, "b c h w -> b h w c")
# print(y.shape)

y0 = x
y1 = reduce(y0, "b c h w -> b c", "max")
y2 = rearrange(y1, "b c -> c b")
y3 = reduce(y2, "c b -> ", "sum")

# print(y1.shape)
# print(y3.shape)

y3.backward()
# print(x.grad)
# print(x.grad[0])
# print(reduce(x.grad, "b c h w -> b", "sum"))

# Flattening
y = rearrange(x, "b c h w -> b (c h w)")
print(y.shape)

# space to depth
y = rearrange(x, "b c (h h1) (w w1) -> b (h1 w1 c) h w", h1 = 2, w1=2)
print(y.shape)


y = reduce(x, "b c h w -> b c", reduction="mean")
print(y.shape)

y = reduce(x, "b c (h h1) (w w1) -> b c h w", reduction="max", h1=2, w1=2)
print(y.shape)

# list_of_tensors = list(x)
# print(f"list of tensors: {list_of_tensors}")

# --------

a = torch.arange(1, 121).reshape(2, 3, 4, 5)
    
#print(a)

b = rearrange(a, "b c h w -> b h c w")
#print(b)

list_of_tensors = list(a)
print(f"list of tensors: {list_of_tensors}")

tensors = rearrange(list_of_tensors, "b c h w -> b h w c")
print(tensors.shape)

tensors2 = rearrange(a, "b c h w -> b h w c")
print(tensors2.shape)

print(tensors)
print(tensors2)

# --------

# Basic implementation

batch = 2
sequence = 3
d_in = 4
d_out = 6
D = torch.arange(batch*sequence*d_in).reshape(batch, sequence, d_in)
A = torch.arange(d_out*d_in).reshape(d_out, d_in)

Y = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")
print(Y)

Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")

print(Y)

# images = torch.randn(64, 128, 128, 3) # (batch, h, w, channel)
images = torch.ones(2, 4, 4, 3) # (batch, h, w, channel)
dim_by = torch.linspace(start=0.0, end=1.0, steps=10)

dim_value = rearrange(dim_by,"dim_value             -> 1 dim_value 1 1 1")
images_rearr = rearrange(images, "b h w c           -> b 1 h w c")
dimmed_images = images_rearr * dim_value

# print(dimmed_images)
dimmed_images = einsum(images, dim_by, "b h w c, dim_value -> b dim_value h w c")
print(dimmed_images)
