# DL- Developing a Recurrent Neural Network Model for Stock Prediction

## AIM
To develop a Recurrent Neural Network (RNN) model for predicting stock prices using historical closing price data.

## Problem Statement and Dataset

Stock price prediction is an important task in financial analysis because investors and organizations rely on accurate forecasts to make better investment decisions. Traditional statistical methods often struggle to capture complex patterns in time-series data such as stock prices.

The objective of this project is to develop a Recurrent Neural Network (RNN) model that can learn patterns from historical stock price data and predict future prices. Using the historical closing prices of Google stock, the model will be trained on a training dataset and evaluated on a separate test dataset.

The system will involve loading the datasets, preprocessing the data, building and training an RNN model, and then predicting stock prices for the test dataset. Finally, the predicted values will be compared with the actual stock prices to evaluate the performance and accuracy of the model.

<img width="599" height="193" alt="image" src="https://github.com/user-attachments/assets/09a90785-f9be-498f-b084-34e3b2999a06" />


## DESIGN STEPS
STEP 1:
Load and normalize data, create sequences.

STEP 2:
Convert data to tensors and set up DataLoader.

STEP 3:
Define the RNN model architecture.

STEP 4:
Summarize, compile with loss and optimizer.

STEP 5:
Train the model with loss tracking.

STEP 6:
Predict on test data, plot actual vs. predicted prices

## PROGRAM

### Name: ALLEN JOVETH P
### Register Number: 212223240007

```python

## Define RNN Model
class RNNModel(nn.Module):
    # write your code here
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
      super(RNNModel, self).__init__()
      self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
      self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
      out,_ = self.rnn(x)
      out = self.fc(out[:,-1,:])
      return out

## Train the Model

# Write your code here
def train_model(model, train_loader, criterion, optimizer, epochs=20):
    train_losses = []
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch =x_batch.to(device),y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        train_losses.append(total_loss / len(train_loader))
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(train_loader):.4f}")
    print('Name: Allen Joveth P')
    print('Register Number:212223240007')
    plt.plot(train_losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training Loss Over Epochs')
    plt.legend()
    plt.show()
train_model(model,train_loader,criterion,optimizer)

```

### OUTPUT

## Training Loss Over Epochs Plot
<img width="567" height="471" alt="image" src="https://github.com/user-attachments/assets/79f5a33c-1d6e-4017-a729-228395b44871" />

## True Stock Price, Predicted Stock Price vs time
<img width="826" height="548" alt="image" src="https://github.com/user-attachments/assets/c5d9df7e-3be5-40d5-a722-619cb2ec4238" />

### Predictions
<img width="224" height="73" alt="image" src="https://github.com/user-attachments/assets/d8fbce82-c984-47f9-b29f-517b03a3f81f" />

## RESULT
Thus, the developing a recurrent neural network model for stock prediction was executed successfully.
