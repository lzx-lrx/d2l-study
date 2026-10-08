import torch
a = torch.arange(6).reshape(2,3)
b = torch.tensor([10,20,30])
print(f"a =\n{a}")
print(f"b =\n{b}")
c = a+b
print(f"c =\n{c}")
x = torch.arange(6).reshape(2, 3)

y1 = torch.tensor([1, 2])                    # 形状 (2,)
y2 = torch.tensor([1, 2]).reshape(2, 1)      # 形状 (2, 1)

print(f"y1.shape = {y1.shape}")
print(f"y2.shape = {y2.shape}")
print(f"x + y1  →  报错（刚才那个）")
print(f"x + y2 =\n{x + y2}")