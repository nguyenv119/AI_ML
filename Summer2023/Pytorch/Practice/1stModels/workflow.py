import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from pathlib import Path

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

# Preparing and loading data --> tensors
# Create KNOWN parameters
weight = 0.7
bias = 0.3
start = 0 
end = 1
step = 0.02
X = torch.arange(start, end, step)

#! The features/parameters for grading: 
'''
    these are represented by weights and bias in the LinearRegressionModel. 
    In the analogy, these would be like your criteria for grading each test 
    (for example, grammar, logic, facts, etc.).
'''
X = X.unsqueeze(dim=1) # unsqueeze so compatible for matmul: weight is [1] and recall we need mxn * n_..
y = weight*X + bias # Linear regression formula


#! Training and test splits from dataset
'''
The test papers you have no knowledge about: 
    These are the data points, represented by the tensors X and y. X represents the features 
    (in this case there's just one feature) and y represents the target or output that 
    you're trying to predict. In this case, y is calculated from X using a known weight 
    and bias. In real world situations, this would be the actual values you're trying to 
    predict (the correct grades on the test papers).

The train and test chunks: 
    These are represented by X_train, y_train (the training set) and X_test, y_test (the testing set). 
    This splitting of data is a common practice in machine learning, where you train your model 
    on a portion of the data, and then test how well it performs on unseen data.
'''
training_split = int(0.8 * len(X))
X_train, y_train = X[:training_split], y[:training_split] # Training split, 60-80% of data
X_test, y_test = X[training_split:], y[training_split:] # Testing split, 10-20% of data



#! Visualize the data with matplotlib
def plot_predictions(train_data=X_train, # x axis
                        train_labels=y_train, # y axis
                        test_data=X_test, # x axis
                        test_labels=y_test, # y axis
                        predictions=None):
    
    #! Plots training data, test data and compares predictions
    plt.figure(figsize=(10, 7))
    # Plot training data in blue, c = color, s = marker size
    plt.scatter(train_data, train_labels, c="b", s=4, label="Training Data")
    # Plot test data in green
    plt.scatter(test_data, test_labels, c="g", s=4, label="Testing Data")

    # Are there predictions?
    if predictions is not None:
            plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")
    # Show the legend
    plt.legend(prop={"size": 14})



#! Start with random weights and biases, look at training data, and adjust through gradient descent
#! & backpropagation to move closer towards the test values
class LinearRegressionModel(nn.Module):

    '''
    #* torch.nn — for building computational graphs
    #* torch.nn.Parameter — parameters nn will learn from
    #* torch.nn.Module — base class for nn modules — overwrite forward method
    #* torch.optim — various algos, optimizers in pytorch, help with grad desc
    #* def forward() - the computation 
    '''
    def __init__(self):
        super().__init__() # initializes the nn.Module, the parent class of LinearRegressionModel

        # Parameters are a special tensor that add itself to the module's parameters when we define a model
        # Changed when we use loss.backward() and optimizer.step()
        self.weights = nn.Parameter(torch.randn(1, #* 1 representing the dimension of tensor
                                               requires_grad=True,
                                               dtype=torch.float))
        self.bias = nn.Parameter(torch.randn(1,
                                            requires_grad=True,
                                            dtype=torch.float))
        
    # Defines the computation to optimize the parameters, used in nn.Module classes. Automatically runs
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weights*x + self.bias # Linear regression


#! Mostly we will create parameters/layers ourselves
class LinearRegressionModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        # Use nn.Linear() since using linear, for creating parameters — inputs and outputs dependant on features of model
        #! Does the y = mx + b and creates correct-features parameters for us in a layer 
        self.linear_layer = nn.Linear(in_features=1,
                                      out_features=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear_layer(x)
        

#! Start grading and comparing your results with the answer key: 
'''
    This is the training loop inside the loop function. 
    For each epoch (or round of grading), you make predictions (y_pred = model(X_train)), 
    compare it to the correct answers to compute the loss (loss = loss_fn(y_pred, y_train)), 
    and then adjust your grading criteria (weights and bias) using backpropagation and 
    gradient descent (loss.backward() and optimizer.step()).
'''
def loop(model, whichModel, X_train, y_train, X_test, y_test, plotLoss, epochs):

    # Setting up the loss function and optimizer
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.SGD(params=model.parameters(), # params to optimize, these are the ones in our model
                                lr=0.01) # learning rate
    #? Track values
    torch.manual_seed(42)
    epoch_count = []
    loss_values = []
    test_loss_values = []
    
    '''
    Repeat this process: 
    The entire training loop is repeated for a number of epochs,
    which is like grading a new batch of test papers each time, and 
    adjusting your grading strategy based on how well you did on the 
    previous batch.
    '''
    #? Training and testing loop
    for epoch in range(epochs): # Loop through data
        model.train() # train mode sets all params that require gradients to require gradients

        #* 1. Forward pass, get the predicted results
        #* 2. Calculate loss
        #* 3. Optimizer zero grad — resets the gradient so doesnt accumulate
        #* 4. Back propagation calculation (partial derivatives) w respect to parameters
        #* Gradient descent (optimizer) — accumulates, which is why we have to step 3

        y_pred = model(X_train)       
        loss = loss_fn(y_pred, y_train)     
        optimizer.zero_grad()      
        loss.backward()     
        optimizer.step()  

        #? Testing Code: test data
        model.eval() # turns off gradient tracking (dropout/batch norm layers)
        with torch.inference_mode():
            test_pred = model(X_test) # 1. forward pass
            test_loss = loss_fn(test_pred, y_test) # 2. Test loss

        if epoch % 10 == 0:
            epoch_count.append(epoch)
            loss_values.append(loss)
            test_loss_values.append(test_loss)

            print(f"Epoch: {epoch} | Loss: {loss} | Test Loss: {test_loss} | Tensors: {model.state_dict()}")

    save_model(model, f"model_{whichModel}") 

    # Plot Loss curves
    if plotLoss:
        plt.plot(epoch_count, np.array(torch.tensor(loss_values).numpy()), label = "Train Loss")
        plt.plot(epoch_count, test_loss_values, label = "Test Loss")
        plt.title("Training and Test Loss Curves")
        plt.ylabel("Loss")
        plt.xlabel("Epochs")
        plt.legend()


def test_model(models, whichModel: int, isNew: bool, isLoop: bool, plotLoss: bool, epochs: int):
    torch.manual_seed(42) #? Control Randomness
    model = models[whichModel]

    #? check if weights file exists and if so, load it
    # if not isNew: load_model(model, "model_0")
    if not isNew: load_model(model, f"model_{whichModel}")

    # Make predictions before optimization. use our model to predict whether X_test matches y_test:
    print(f"Original: {model.state_dict()}\n\n")
    with torch.inference_mode(): 
        y_preds = model(X_test) 
    plot_predictions(predictions=y_preds)
    plt.show()

    if isLoop: loop(model, whichModel, X_train, y_train, X_test, y_test, plotLoss, epochs)
     
    # Make predictions after optimization
    print(f"After Train: {model.state_dict()}\n\n")
    with torch.inference_mode():
        y_preds = model(X_test)
    plot_predictions(predictions=y_preds)
    plt.show() 


torch.manual_seed(42)
models = {0: LinearRegressionModel(),
          1: LinearRegressionModelV2()}
plt.show()
test_model(models=models, whichModel=1, isNew=False, isLoop=True, plotLoss=True, epochs=200)

'''
So, in summary, the goal of this code (and of training any machine learning model) 
is to "learn" the best set of parameters (weights and bias) that would make the model's
predictions as close as possible to the actual target values. In the test grading analogy, 
this is like refining your grading strategy until your grades match as closely as possible
with the answer key.
'''
