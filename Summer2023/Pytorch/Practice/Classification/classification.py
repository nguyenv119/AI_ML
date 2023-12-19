import torch
from torch import nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

import sklearn 
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd

import requests # get code online

if Path("helper_functions.py").is_file():
    print("\nhelper_functions.py already exists, skipping download\n")
else: 
    print("\nDownload helper_functions.py")
    request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
    with open("helper_functions.py", "wb") as f: # open this file and write, wb
        f.write(request.content)


#* Calculate accuracy: out of 100 examples, what % does model predict right
from helper_functions import accuracy_fn, plot_predictions, plot_decision_boundary

# Save mode's parameters' into model.pth, in pickle format
# torch.save saves a Pytorch object
def save_model(model, model_name):
    MODEL_DIR = Path('Models') # Define the directory where you want to save your models
    MODEL_DIR.mkdir(exist_ok=True) # Create the directory if it doesn't exist
    MODEL_PATH = MODEL_DIR / f"{model_name}.pth" # Define the path to the new model file
    torch.save(model.state_dict(), MODEL_PATH) # Save the model
def load_model(model, model_name):
    MODEL_DIR = Path('Models') # Define the directory where your models are saved
    MODEL_PATH = MODEL_DIR / f"{model_name}.pth" # Define the path to the model file
    if MODEL_PATH.exists(): # Check if the model file exists
        model.load_state_dict(torch.load(MODEL_PATH)) # Load the model
    else:
        print(f"No model found at {MODEL_PATH}")

#! Predicting whether smth is one thing or another: binary (1 or the other)/multiclass(many things, multiple ourput nodes) classification

#? Create data
n_samples = 1000

#* 2D features and labels (0, 1)
X, y = make_circles(n_samples,
                    noise=0.03, # add randomenss, standard deviation
                    random_state=42) # same as random seed

# circles = pd.DataFrame({"X1": X[:, 0], # 0th index
#                         "X2": X[:, 1], # 1th index 
#                         "label": y})
# print(f"{circles.head(10)}\n")
# plt.figure(figsize=(10, 7))
# plt.scatter(x=X[:, 0],
#             y=X[:, 1],
#             s=4,
#             c=y, # dependant on which class 
#             cmap=plt.cm.RdYlBu)

# (1000, 2) 1000 samples, 2 features
# (1000,) 1000 samples, 0 features

# ? Turn data into tensors, since its from numpy and dtype float64. We need float32
X = torch.from_numpy(X).type(torch.float32)
y = torch.from_numpy(y).type(torch.float32)
print(f"X Shape: {X.shape}, Y Shape: {y.shape}, and both are of dtype: {X.dtype}\n")

#? Split into training and test sets
X_train, X_test, y_true_train, y_true_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2,
                                                    random_state=42) # 20% of data will be test data


# #? Building the model to classify blue red dots
class CircleModelV0(nn.Module):
     def __init__(self):
        super().__init__()
    
        #* Method 1: Create 2 linear layers, more out_features, more chances for network to detect patterns
        self.layer_1 = nn.Linear(in_features=2, out_features=5)
        self.layer_2 = nn.Linear(in_features=5, out_features=1)

        #* Method 2: nn.Sequential
        # self.two_linear_layers = nn.Sequential(
        #     nn.Linear(in_featuers=2, out_features=5),
        #     nn.Linear(in_features=5, out_features=1)
        # )

    #*x (input) -> layer 1 -> layer 2 -> output
     def forward(self, x):
        #  return two_linear_layers(x)
         return self.layer_2(self.layer_1(x))


# increasing hidden units, hidden layers, and epochs
class CircleModelV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
    
    def forward(self, x):
        return self.layer_3(self.layer_2(self.layer_1(x)))

#? nonlinear model
class CircleModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=10)
        self.layer_3 = nn.Linear(in_features=10, out_features=1)
        self.relu = nn.ReLU() # nonlinear actv function

    def forward(self, x): # x -> layer1 -> relu -> layer2 -> relu -> layer3
        return self.layer_3(self.relu(self.layer_2(self.relu(self.layer_1(x)))))
    

# #* each row represents connections from input to hidden layer, and each element in the index represent the wieght represented by each input node
# For example, 0.6580 is the 1st weight from 1st input node to 1st node of hidden layer, and 0.2234 is 2nd node of input layer connecting to 1st node of hidden layer

# model_2 = CircleModelV1()
model_3 = CircleModelV2()

with torch.inference_mode():
    untrained_preds = model_3(X_test)
    print(f"Shape of predictions: {untrained_preds.shape}")
    print(f"Shape of test samples: {y_true_test.shape}")
    print(f"\nFirst 10 predictions:\n{untrained_preds[:10]}")
    print(f"\nFirst 10 test labels:\n{y_true_test[:10]}\n")

print(model_3)

#! Set up training loop
# loss_fn = nn.BCELoss() # requires inputs to have gone through sigmoid function prior to input
# y_pred = loss_fn(torch.sigmoid(y_logits), y_true_train)

loss_fn = nn.BCEWithLogitsLoss() # combines sigmoid and BCELoss (multiclass classification use softmax).
optimizer = torch.optim.SGD(params=model_3.parameters(),
                            lr = 0.1)

torch.manual_seed(42)
epochs = 1000
for epoch in range(epochs):

    #? Training with X_train
    model_3.train()

    # print(f"See how each element of X_train is {X_train[0]} and shape {X_train[0].shape}")
    # print(f"y_true_train: {y_true_train[0]} and shape {y_true_train[0].shape}")
    # y_logit = model_3(X_train)[0]
    # print(f"Output logit is {y_logit} and shape {y_logit.shape}. Need to squeeze to have same tensor dims for loss_fn")
    # print(f"Squeezed: {y_logit.squeeze()} and shape: {y_logit.squeeze().shape}")
 
    y_logits = model_3(X_train).squeeze() # Remember to squeeze since BCE requires tensors to be of same shape
    y_pred = torch.round(torch.sigmoid(y_logits))  # decision boundary: y_pred_probs > 0.5 ? 1 : 0
    loss = loss_fn(y_logits, y_true_train) # Remember BCELosswithLogis expects raw logits
    acc = accuracy_fn(y_true=y_true_train,
                      y_pred=y_pred)
    
    optimizer.zero_grad() 
    loss.backward() # backpropagation
    optimizer.step() # gradient descent

    #? Testing with X_test
    #* TODO: raw logits(outputs) -> prediction prob(activation functions) -> prediction labels(round or argmax)
    model_3.eval()
    with torch.inference_mode():
        test_logits = model_3(X_test).squeeze()
        test_preds = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_true_test)
        test_acc = accuracy_fn(y_true=y_true_test,
                               y_pred=test_preds)
        
    if epoch % 100 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")

#? Visualize preds: looks like its good as guessing, makes sense since 2 labels, 50% guessing

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_3, X_train, y_true_train)

plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_3, X_test, y_true_test)
plt.show()







# weight, bias, start, end, step = 0.7, 0.3, 0, 1, 0.01
# X_regression = torch.arange(start, end, step).unsqueeze(dim=1)
# y_regression = weight * X_regression + bias

# train_split = int(0.8 * len(X_regression))
# X_train_regression, y_train_regression = X_regression[:train_split], y_regression[:train_split]
# X_test_regression, y_test_regression = X_regression[train_split:], y_regression[train_split:]

# plot_predictions(train_data=X_train_regression,
#                  train_labels=y_train_regression,
#                  test_data=X_test_regression,
#                  test_labels=y_test_regression)
# plt.title("Testing")
# # plt.show()

# model_2 = nn.Sequential(
#     nn.Linear(in_features=1, out_features=10),
#     nn.Linear(in_features=10, out_features=10),
#     nn.Linear(in_features=10, out_features=1)
# )

# loss_fn = nn.L1Loss()
# optimizer = torch.optim.SGD(params=model_2.parameters(),
#                            lr=0.01)

# torch.manual_seed(42)
# epochs = 1000
# for epoch in range(epochs):
#     model_2.train()
#     y_pred = model_2(X_train_regression)
#     loss = loss_fn(y_pred, y_train_regression)
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()

#     model_2.eval()
#     with torch.inference_mode():
#         test_pred = model_2(X_test_regression)
#         test_loss = loss_fn(test_pred, y_test_regression)
    

#     # No need accuracy since working with regression, not classification
#     if epoch % 100 == 0:
#         print(f"Epoch: {epoch} | Loss: {loss:.5f} | Test loss: {test_loss:.5f}")


# model_2.eval()
# with torch.inference_mode():
#     test_pred = model_2(X_test_regression)
    
# plot_predictions(train_data=X_train_regression,
#                 train_labels=y_train_regression,
#                 test_data=X_test_regression,
#                 test_labels=y_test_regression,
#                 predictions=test_pred)