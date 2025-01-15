import torch

x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(0.5, requires_grad=True)
w1 = torch.tensor(-3.0, requires_grad=True)
w2 = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(6.8, requires_grad=True)

mul1 = x1 * w1
mul2 = x2 * w2
add1 = mul1 + mul2
add2 = add1 + b
output = torch.tanh(add2)

output.backward()

print(output)
print("x1.grad", x1.grad)
print("x2.grad", x2.grad)
print("w1.grad", w1.grad)
print("w2.grad", w2.grad)
print("b.grad", b.grad)
