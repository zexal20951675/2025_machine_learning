# Week 3 Programming Assignment: Approximate f(x) and f'(x) using neural network
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

# 1. Generate training data and derivative data
x_train = np.linspace(-1, 1, 100).reshape(-1, 1)
y_train = 1 / (1 + 25 * x_train**2)
dy_train = (-50 * x_train) / (1 + 25 * x_train**2)**2

x_train_tensor = torch.tensor(x_train, dtype=torch.float32, requires_grad=True)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
dy_train_tensor = torch.tensor(dy_train, dtype=torch.float32)

# 2. Define the neural network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 1)
        self.activation = nn.Tanh()

    def forward(self, x):
        x = self.activation(self.fc1(x))
        x = self.activation(self.fc2(x))
        x = self.fc3(x)
        return x

net = Net()

# 3. Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.01)

# 4. Train the network (joint loss: function + derivative)
num_epochs = 300
loss_history = []
lambda_derivative = 1.0  # weight for derivative loss

for epoch in range(num_epochs):
    optimizer.zero_grad()
    output = net(x_train_tensor)

    # Loss on function values
    loss_f = criterion(output, y_train_tensor)

    # Loss on derivatives using autograd
    grads = torch.autograd.grad(outputs=output,
                                inputs=x_train_tensor,
                                grad_outputs=torch.ones_like(output),
                                create_graph=True)[0]
    loss_df = criterion(grads, dy_train_tensor)

    # Total loss
    loss = loss_f + lambda_derivative * loss_df

    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        loss_history.append(loss.item())
        print(f"Epoch {epoch}: Total Loss = {loss.item():.6f}, f Loss = {loss_f.item():.6f}, f' Loss = {loss_df.item():.6f}")

# 5. Evaluate network on test set
x_test = np.linspace(-1, 1, 200).reshape(-1, 1)
y_true = 1 / (1 + 25 * x_test**2)
dy_true = (-50 * x_test) / (1 + 25 * x_test**2)**2

x_test_tensor = torch.tensor(x_test, dtype=torch.float32, requires_grad=True)
y_pred = net(x_test_tensor)
dy_pred = torch.autograd.grad(outputs=y_pred,
                              inputs=x_test_tensor,
                              grad_outputs=torch.ones_like(y_pred),
                              create_graph=False)[0]

# Convert predictions to NumPy
y_pred_np = y_pred.detach().numpy()
dy_pred_np = dy_pred.detach().numpy()

# 6. Plot function approximation
plt.figure(figsize=(8, 5))
plt.plot(x_test, y_true, label="True f(x)")
plt.plot(x_test, y_pred_np, label="NN f(x) Prediction")
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Function Approximation")
plt.legend()
plt.grid(True)
plt.show()

# 7. Plot derivative approximation
plt.figure(figsize=(8, 5))
plt.plot(x_test, dy_true, label="True f'(x)")
plt.plot(x_test, dy_pred_np, label="NN f'(x) Prediction")
plt.xlabel("x")
plt.ylabel("f'(x)")
plt.title("Derivative Approximation")
plt.legend()
plt.grid(True)
plt.show()

# 8. Plot training loss
epochs_recorded = np.arange(0, num_epochs, 5)
plt.figure(figsize=(8, 5))
plt.plot(epochs_recorded, loss_history, marker='o')
plt.xlabel("Epoch")
plt.ylabel("Total Loss")
plt.title("Training Loss Curve (Every 5 Epochs)")
plt.grid(True)
plt.show()

# 9. Report errors
mse_f = np.mean((y_pred_np - y_true)**2)
mse_df = np.mean((dy_pred_np - dy_true)**2)
max_err_f = np.max(np.abs(y_pred_np - y_true))
max_err_df = np.max(np.abs(dy_pred_np - dy_true))

print("MSE (Function):", mse_f)
print("MSE (Derivative):", mse_df)
print("Max Error (Function):", max_err_f)
print("Max Error (Derivative):", max_err_df)
