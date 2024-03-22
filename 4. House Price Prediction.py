import numpy as np
import pandas as pd
import datetime
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt 

'''
What we will learn -
1. Theoritical knowledge of Deep Learning
2. ANN
3. Feature Engineering (Categorical --> Embedding layer, Continous Variable)
4. Pythonic Class to Create Feed Forward Neural Networks

Dataset --> Features {Categorical, Continous}
Pytorch -- Tabular Dataset

1. Categorical Features -- Embedding Layers
2. Continous Features

1. Categorical Features --
(a) Label Encoding
(b) Take all categorical features --> {numpy, torch-->tensors}
(c) Lates take all the continous values
(d) Continous variable --> {numpy, torch-->tensors}
(e) Embedding Layers --> Categorical Features



'''


class FeedForwardNN(nn.Module):
    def __init__(self, embedding_dim, n_cont, out_sz, layers, p=0.5):
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(inp, out) for inp, out in embedding_dim])
        self.emb_drop = nn.Dropout(p)
        self.bn_cont = nn.BatchNorm1d(n_cont)

        layerlist = []
        n_emb = sum(out for inp, out in embedding_dim)
        n_in = n_emb + n_cont

        for i in layers:
            layerlist.append(nn.Linear(n_in, i))
            layerlist.append(nn.ReLU(inplace=True))
            layerlist.append(nn.BatchNorm1d(i))
            layerlist.append(nn.Dropout(p))
            n_in = i
        layerlist.append(nn.Linear(layers[-1], out_sz))
        self.layers = nn.Sequential(*layerlist)
    
    def forward(self, x_cat, x_cont):
        embeddings = []
        for i, e in enumerate(self.embeds):
            embeddings.append(e(x_cat[:, i]))
        x = torch.cat(embeddings, 1)
        x = self.emb_drop(x)

        x_cont = self.bn_cont(x_cont)
        x = torch.cat([x, x_cont], 1)
        x = self.layers(x)
        return x

df = pd.read_csv('Datasets/house-prices-advanced-regression-techniques/train.csv', usecols=['SalePrice', 'MSSubClass', 'MSZoning', 'LotFrontage', 'LotArea', 'Street', 'YearBuilt', 'LotShape', '1stFlrSF', '2ndFlrSF']).dropna()

def df_info():
    print(df.shape)
    print(df.head())
    print(df.info())

    
def do_task():
    ### Data Pre-Processing

    for i in df.columns:
        print(f'Column name {i} and unique values are {len(df[i].unique())}')

    current_year = datetime.datetime.now().year
    print(current_year)

    df['Total Years'] = current_year - df['YearBuilt']
    print(df.head())
    df.drop('YearBuilt', axis=1, inplace=True)

    # Define categorical features
    cat_features = ['MSSubClass', 'MSZoning', 'Street', 'LotShape']
    out_features = ['SalePrice']

    # Encoding my categorical values
    lbl_encoders = {}
    
    for feature in cat_features:
        lbl_encoders[feature] = LabelEncoder()
        df[feature] = lbl_encoders[feature].fit_transform(df[feature])
    
    print(df.head())

    # Stacking and converting into Tensors (numpy -> tensor)
    cat_features = np.stack([df['MSSubClass'], df['MSZoning'], df['Street'], df['LotShape']], 1)
    print(cat_features)

    # Convert numpy to tensors
    cat_features = torch.tensor(cat_features, dtype=torch.int64) # Categorical features can't be converted into float.
    print(cat_features)

    # Create continous variable
    cont_features = []
    for i in df.columns:
        if i not in ['MSSubClass', 'MSZoning', 'Street', 'LotShape', 'SalePrice']:
            cont_features.append(i)

    print(cont_features)

    # Stacking continous vaiable to a tensor
    cont_values = np.stack([df[i].values for i in cont_features], axis=1)
    cont_values = torch.tensor(cont_values, dtype=torch.float)
    print(cont_values)
    print(cont_values.dtype)

    # Dependent Features
    y = torch.tensor(df['SalePrice'].values, dtype=torch.float).reshape(-1, 1)  # We require a 2D tensor, always do the reshape!
    print(y)
    print(y.shape)

    print(f'All features shape: {cat_features.shape, cont_values.shape, y.shape}')


    ### Embedding Size for Categorical Columns
    print(len(df['MSSubClass'].unique()))
    print(len(df['MSZoning'].unique()))
    print(len(df['Street'].unique()))
    print(len(df['LotShape'].unique()))

    # Why dimension is required in embedding?
    # This length is important! This length will say that, my embedding layer how many inputs it should have and based on that how many outputs should I create.

    cat_dims = [len(df[i].unique()) for i in ['MSSubClass', 'MSZoning', 'Street', 'LotShape']]
    print(cat_dims)

    ### Netural Netwrok

    ### Thumb Rules: Output dimension should be set based on the input dimension (min(50, feature dimension (unique values) / 2))
    ## Docs Link: https://docs.fast.ai/tabular.html
    ## https://docs.fast.ai/2018/04/29/categorical-embeddings/

    embedding_dim = [(x, min(50, (x + 1) // 2)) for x in cat_dims]
    print(embedding_dim)

    embed_representation = nn.ModuleList([nn.Embedding(inp, out) for inp, out in embedding_dim])
    print(embed_representation)

    print(cat_features)

    cat_featuresz = cat_features[:4]
    print(cat_featuresz)

    # Convert cat_featuresz into vectors with the help of embedding technique
    embedding_val = []
    for i, e in enumerate(embed_representation):
        embedding_val.append(e(cat_features[:, i]))
    print("Embedding value: ")
    print(embedding_val)
    
    # But, here, the stacking is not in a proper way. We should stacked in column wise.
    z = torch.cat(embedding_val, 1)
    print(z)

    # Implement Dropout
    dropout = nn.Dropout(0.4)

    final_embed = dropout(z)
    print(final_embed)


    # Run NN
    torch.manual_seed(100)
    model = FeedForwardNN(embedding_dim, len(cont_features), 1, [100, 50], p=0.4)
    print("Model: ")
    print(model)

    ## Define loss and optimizer
    loss_funtion = nn.MSELoss()  ## later convert to RMSE
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    batch_size = 1200
    test_size = int(batch_size * 0.15)
    train_categorical = cat_features[:batch_size - test_size]
    test_categorical = cat_features[batch_size - test_size:batch_size]
    train_cont = cont_values[:batch_size - test_size]
    test_cont = cont_values[batch_size - test_size:batch_size]

    y_train = y[:batch_size - test_size]
    y_test = y[batch_size - test_size:batch_size]

    print(len(train_categorical), len(test_categorical), len(train_cont), len(test_cont), len(y_train), len(y_test))

    epochs = 5000
    final_losses = []
    for i in range(epochs):
        i = i + 1
        y_pred = model(train_categorical, train_cont)
        loss = torch.sqrt(loss_funtion(y_pred, y_train)) ### RMSE
        final_losses.append(loss)
        if i % 10 == 1:
            print(f"Epoch number: {i}, and the loss: {loss.item()}")
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        plt.plot(range(epochs), final_losses)
        plt.ylabel('RMSE Loss')
        plt.xlabel('Epochs')
        plt.show() 

    ## Validate the Test Data
    y_pred = ""
    with torch.no_grad():
        y_pred = model(test_categorical, test_cont)
        loss = torch.sqrt(loss_funtion(y_pred, y_test))
    print(f'RMSE: {loss}')

    data_verify = pd.DataFrame(y_test.tolist(), columns=['Test'])
    data_predicted = pd.DataFrame(y_pred.tolist(), columns=['Prediction'])

    final_output = pd.concat([data_verify, data_predicted], axis=1)
    final_output['Difference'] = final_output['Test'] - final_output['Prediction']
    print(final_output.head())

    # Saving the model
    torch.save(model, 'HousePrice.pt')
    torch.save(model.state_dict(), 'HouseWeights.pt')

    # Loading the saved model
    embs_size = [(15, 8), (5, 3), (2, 1), (4, 2)]
    model1 = FeedForwardNN(embs_size, 5, 1, [100, 50], p=0.4)
    model1.load_state_dict(torch.load('D:\Github Repos\PyTorch\HouseWeights.pt'))

    print(model1.eval())

if __name__ == '__main__':
    df_info()
    do_task()