import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Load stock price data (example CSV)
def load_data(file_path):
    df = pd.read_csv(file_path)
    return df[['Close']].values  # Assuming 'Close' is the stock price


# Preprocess data (normalize and create sequences for time series)
def prepare_data(data, seq_length):
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_normalized = scaler.fit_transform(data)

    x, y = [], []
    for i in range(len(data_normalized) - seq_length):
        x.append(data_normalized[i:i + seq_length])
        y.append(data_normalized[i + seq_length])

    return np.array(x), np.array(y), scaler


# Define the model
class StockPredictor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(StockPredictor, self).__init__()
        self.fc1 = nn.Linear(input_size * 1, hidden_size)  # Flatten input
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # Flatten the input from (batch_size, seq_length, 1) to (batch_size, seq_length)
        x = x.view(x.size(0), -1)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out


# Train the model
def train_model(model, criterion, optimizer, train_loader, epochs=50):
    for epoch in range(epochs):
        for inputs, targets in train_loader:
            inputs, targets = inputs.float(), targets.float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        if epoch % 2 == 0:
            print(f'Epoch {epoch}/{epochs}, Loss: {loss.item()}')


# Evaluate the model
def evaluate_model(model, test_loader, criterion, scaler):
    model.eval()  # Set model to evaluation mode
    total_loss = 0
    predictions, actuals = [], []

    with torch.no_grad():  # No need to track gradients during evaluation
        for inputs, targets in test_loader:
            inputs, targets = inputs.float(), targets.float()
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, targets)
            total_loss += loss.item()

            # Append normalized predictions and actuals
            predictions.append(outputs.numpy())
            actuals.append(targets.numpy())

    # Invert normalization
    predictions = scaler.inverse_transform(np.concatenate(predictions))
    actuals = scaler.inverse_transform(np.concatenate(actuals))

    # Compute average loss
    avg_loss = total_loss / len(test_loader)
    print(f"Evaluation Loss: {avg_loss:.4f}")

    return predictions, actuals


def plot_predictions(predictions, actuals, title='Predicted vs Actual Values'):
    # Convert the list of arrays to a single numpy array
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)

    # Create a plot
    plt.figure(figsize=(10, 6))
    plt.plot(actuals, label='Actual Values', color='blue')
    plt.plot(predictions, label='Predicted Values', color='red', linestyle='--')

    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Stock Price')
    plt.legend()
    plt.show()


# Save the model to the 'models/' directory
def save_model(model, filename):
    # Ensure the 'models/' directory exists
    if not os.path.exists('models'):
        os.makedirs('models')

    # Save the model in 'models/' directory
    file_path = os.path.join('models', filename)
    torch.save(model.state_dict(), file_path)
    print(f"Model saved to {file_path}")


# Load the model from the 'models/' directory
def load_model(model, filename):
    file_path = os.path.join('models', filename)

    # Check if the file exists before loading
    if os.path.exists(file_path):
        model.load_state_dict(torch.load(file_path, weights_only=True))
        model.eval()  # Set the model to evaluation mode after loading
        print(f"Model loaded from {file_path}")
    else:
        print(f"Error: {file_path} does not exist.")


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or load a stock prediction model")
    parser.add_argument('--save', type=str, help="File path to save the model")
    parser.add_argument('--load', type=str, help="File path to load a pre-trained model")
    args = parser.parse_args()

    # Load and prepare data
    data = load_data('data/SPX_1min_sample.csv')
    seq_length = 60  # Use past 60 days to predict the next day
    x, y, scaler = prepare_data(data, seq_length)

    # Train/test split
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, shuffle=False)

    # Convert to tensors
    train_data = torch.utils.data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test_data = torch.utils.data.TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=32)

    # Model parameters
    input_size = seq_length
    hidden_size = 128
    output_size = 1
    model = StockPredictor(input_size, hidden_size, output_size)

    # Load model if specified
    if args.load:
        load_model(model, args.load)

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model if not loading
    if not args.load:
        train_model(model, criterion, optimizer, train_loader, epochs=100)

    # Save the model if specified
    if args.save:
        save_model(model, args.save)

    # Evaluate the model
    predictions, actuals = evaluate_model(model, test_loader, criterion, scaler)

    plot_predictions(predictions, actuals)
