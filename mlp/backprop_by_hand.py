"""
Backpropagation by hand, with plain numbers: no matrices, no loops.

A tiny network with 2 inputs, 1 hidden neuron and 1 output:

    z1   = w1*x1 + w2*x2 + b1        (weighted sum)
    h    = tanh(z1)                  (activation)
    yhat = w3*h + b2                 (output)
    L    = 0.5 * (yhat - y)^2        (loss: how wrong we are)

We compute every derivative with the chain rule, check it numerically, and
take one gradient-descent step to show the loss going down.

Run:
    python mlp/backprop_by_hand.py
"""
import math

x1, x2, y = 1.0, -2.0, 0.5  # one training example and its target
params = {"w1": 0.3, "w2": -0.1, "b1": 0.0, "w3": 0.8, "b2": 0.1}


def forward(p, verbose=False):
    z1 = p["w1"] * x1 + p["w2"] * x2 + p["b1"]
    h = math.tanh(z1)
    yhat = p["w3"] * h + p["b2"]
    L = 0.5 * (yhat - y) ** 2
    if verbose:
        print("FORWARD PASS (compute the prediction and the loss)")
        print(f"  z1   = w1*x1 + w2*x2 + b1 = {p['w1']}*{x1} + {p['w2']}*{x2} + {p['b1']} = {z1:.4f}")
        print(f"  h    = tanh(z1)           = {h:.4f}")
        print(f"  yhat = w3*h + b2          = {p['w3']}*{h:.4f} + {p['b2']} = {yhat:.4f}")
        print(f"  L    = 0.5*(yhat - y)^2   = 0.5*({yhat:.4f} - {y})^2 = {L:.4f}\n")
    return z1, h, yhat, L


p = dict(params)
z1, h, yhat, L = forward(p, verbose=True)

print("BACKWARD PASS (chain rule, from the loss back to each weight)")
dL_dyhat = yhat - y
print(f"  dL/dyhat = yhat - y                 = {dL_dyhat:.4f}")
dL_dw3 = dL_dyhat * h
dL_db2 = dL_dyhat * 1.0
print(f"  dL/dw3   = dL/dyhat * h             = {dL_dw3:.4f}")
print(f"  dL/db2   = dL/dyhat * 1             = {dL_db2:.4f}")
dL_dh = dL_dyhat * p["w3"]
print(f"  dL/dh    = dL/dyhat * w3            = {dL_dh:.4f}")
dL_dz1 = dL_dh * (1 - h ** 2)
print(f"  dL/dz1   = dL/dh * (1 - h^2)        = {dL_dz1:.4f}   (tanh' = 1 - tanh^2)")
dL_dw1 = dL_dz1 * x1
dL_dw2 = dL_dz1 * x2
dL_db1 = dL_dz1 * 1.0
print(f"  dL/dw1   = dL/dz1 * x1              = {dL_dw1:.4f}")
print(f"  dL/dw2   = dL/dz1 * x2              = {dL_dw2:.4f}")
print(f"  dL/db1   = dL/dz1 * 1               = {dL_db1:.4f}\n")
grads = {"w1": dL_dw1, "w2": dL_dw2, "b1": dL_db1, "w3": dL_dw3, "b2": dL_db2}

print("CHECK: nudge each weight a tiny bit and see how much L changes")
eps = 1e-6
for k in params:
    plus, minus = dict(p), dict(p)
    plus[k] += eps
    minus[k] -= eps
    numeric = (forward(plus)[3] - forward(minus)[3]) / (2 * eps)
    print(f"  {k}: chain rule {grads[k]: .6f}   numerical {numeric: .6f}")

lr = 0.5
print(f"\nGRADIENT DESCENT STEP (lr = {lr}): w = w - lr * dL/dw")
for k in p:
    new = p[k] - lr * grads[k]
    print(f"  {k}: {p[k]: .4f} -> {new: .4f}")
    p[k] = new
_, _, new_yhat, new_L = forward(p)
print(f"\nprediction {yhat:.4f} -> {new_yhat:.4f} (target {y})")
print(f"loss       {L:.4f} -> {new_L:.4f}   the loss went down, so the network learned a bit.")
