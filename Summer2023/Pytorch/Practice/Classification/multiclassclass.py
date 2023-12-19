import torch
from torch import nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 

import sklearn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from helper_functions import accuracy_fn, plot_predictions, plot_decision_boundary

# Set hyperparams for datasets
NUM_CLASSES = 4
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42
EPOCHS = 100
LEARNING_RATE = 0.1

# 1. CREATE MULTICLASS DATA: https://scikit-learn.org/stable/modules/generated/sklearn.datasets.make_blobs.html
X_blob, y_blob = make_blobs(n_samples=1000,
                             n_features=NUM_FEATURES,
                            centers=NUM_CLASSES,
                            cluster_std=1.5,
                            random_state=RANDOM_SEED)

# 2. TURN DATA INTO TENSORS and split train test: https://pytorch.org/docs/stable/generated/torch.from_numpy.html
X_blob = torch.from_numpy(X_blob).type(torch.float32)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor) # need longTensor 

X_blob_train, X_blob_test, y_blob_train, y_blob_test = train_test_split(X_blob, # will be shape of (batchsize, # features)
                                                                        y_blob, # will be shape of (batchsize), determining using argmax, which class it is likely to be
                                                                        test_size=0.2,
                                                                        random_state=RANDOM_SEED)

# 3. Visualize data
plt.figure(figsize=(10, 7))
plt.scatter(X_blob[:, 0], X_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu)
plt.show()

# print(X_blob.shape, y_blob.shape) X has 2 features (input node features), and y is a scalar (means we have to squeeze later to match y_blob_preds)
# https://pytorch.org/docs/stable/generated/torch.unique.html
# print(torch.unique(y_blob_train)) shows that y has 4 unique values, 4 different classes
# plt.show()

# 4. Building multiclass classification model
class BlobModel(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=8):
        super().__init__()
        # passes through sequentially
        self.linear_layer_stack = nn.Sequential(
            nn.Linear(in_features=input_features, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units, out_features=output_features)
        )
    
    def forward(self, x):
        return self.linear_layer_stack(x)

torch.manual_seed(RANDOM_SEED)
model_4 = BlobModel(input_features=2, output_features=4)

# 5. Create loss fn, optimizer, and training loop
loss_fn = nn.CrossEntropyLoss() # the params are not same shape here since its multiclass classification
optimizer = torch.optim.SGD(params=model_4.parameters(), 
                            lr=LEARNING_RATE)

# y_logits = model_4(X_blob_train).squeeze()
# y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)

torch.manual_seed(RANDOM_SEED)
for epoch in range(EPOCHS):
    model_4.train()
    # need to convert logits(model) -> preds(softmax) -> labels(argmax)
    y_logits = model_4(X_blob_train)

    # https://pytorch.org/docs/stable/generated/torch.nn.Softmax.html
    # Softmax performs the operations on the given dimensions. Here, we want the rows to sum to 1, probability, and the rows here are dim=1. If dim=0, then the columns would sum since columns here are dim=0.
    y_preds = torch.softmax(y_logits, dim=1).argmax(dim=1)

    loss = loss_fn(y_logits, y_blob_train) #! internally applies softmax and does CE loss. shapes not the same since mutliclass classification
    acc = accuracy_fn(y_true=y_blob_train,
                      y_pred=y_preds)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model_4.eval()
    with torch.inference_mode():
        test_logits = model_4(X_blob_test)
        test_preds = torch.softmax(test_logits, dim=1).argmax(dim=1)

        test_loss = loss_fn(test_logits, y_blob_test)
        test_acc = accuracy_fn(y_true=y_blob_test,
                               y_pred=test_preds)
        
    if epoch % (EPOCHS / 10) == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Acc: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")
