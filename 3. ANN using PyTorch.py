import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score

# Libraries from PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F 


df = pd.read_csv('D:\Github Repos\PyTorch\Datasets\diabetes.csv')

def plot_dataset_info():
    print(df.head())
    print(df.isnull().sum())
    print(df.shape)

    ## Example for showing a random example of seaborn
    # df['Outcome'] = np.where(df['Outcome'] == 1, "Diabetic", "No Diabetic")
    # sns.pairplot(df, hue='Outcome')
    # plt.show()

def split_data(df, test_size=0.2):
    X = df.drop('Outcome', axis=1).values
    y = df['Outcome'].to_numpy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    return X_train, y_train, X_test, y_test


## Create model using PyTorch
### Use categorical cross entropy for the loss function instead of binary cross entropy to show that how the probability values will be comming up.
class ANN_Model(nn.Module):
    def __init__(self, input_features=8, hidden1=20, hidden2=20, out_features=2):
        super().__init__()
        self.f_connected1 = nn.Linear(input_features, hidden1)
        self.f_connected2 = nn.Linear(hidden1, hidden2)
        self.out = nn.Linear(hidden2, out_features)

    def forward(self, x):
        x = F.relu(self.f_connected1(x))
        x = F.relu(self.f_connected2(x))
        x = self.out(x)
        return x


if __name__ == '__main__':
    plot_dataset_info()
    X_train, y_train, X_test, y_test = split_data(df)

    # Create tensors
    X_train = torch.FloatTensor(X_train) # independent features needs to be converted into FloatTensor
    X_test = torch.FloatTensor(X_test)

    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    # Instantiate my ANN_Model
    torch.manual_seed(20)
    model = ANN_Model()
    print(model)
    print(model.parameters)

    # Backward Propagation --- Define the loss_function, define the optimizer
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    epochs = 500
    final_losses = []

    for i in range(epochs):
        i = i + 1
        y_pred = model.forward(X_train)
        loss = loss_function(y_pred, y_train)
        final_losses.append(loss)
        
        if i % 10 == 1:
            print(f'Epoch number is {i} and the loss is {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():

        # Plot the loss function
        plt.plot(range(epochs), final_losses)
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()
        
        # Prediction in X_test data
        predictions = []
        for i, data in enumerate(X_test):
            y_pred = model(data)
            predictions.append(y_pred.argmax().item())
            print(y_pred.argmax().item())

        cm = confusion_matrix(y_test, predictions)
        print(cm)

        plt.figure(figsize=(10, 6))
        sns.heatmap(cm, annot=True)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.show()

        score = accuracy_score(y_test, predictions)
        print(f'Accuracy: {score}')

    # Save the model
    torch.save(model, 'diabetes.pt')

    # Load the model
    model = torch.load('diabetes.pt')
    print(model.eval())

    # Prediction of new data point
    print(list(df.iloc[0, :-1]))

    # New data
    lst1 = [6.0, 130.0, 72.0, 40.0, 0.0, 25.6, 0.627, 45.0]
    new_data = torch.tensor(lst1)

    # Predict new data using PyTorch
    with torch.no_grad():
        print(model(new_data))
        print(model(new_data).argmax().item())

    