import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

class SpamDataset(Dataset):
    def __init__(self, X, y):
        # Convert to numpy arrays first if they're pandas Series/DataFrame
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.to_numpy()
        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.to_numpy()
            
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y.reshape(-1, 1))  # Reshape to 2D array
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

class SpamClassifier(nn.Module):
    def __init__(self, input_dim):
        super(SpamClassifier, self).__init__()
        self.layer1 = nn.Linear(input_dim, 64)
        self.layer2 = nn.Linear(64, 32)
        self.layer3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.3)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        x = self.dropout(self.relu(self.layer1(x)))
        x = self.dropout(self.relu(self.layer2(x)))
        x = self.sigmoid(self.layer3(x))
        return x

def load_and_prepare_data(file_path="spam_call_dataset.csv"):
    # Load the dataset
    df = pd.read_csv(file_path)
    
    # Prepare features
    df['Time_Encoded'] = pd.Categorical(df['Time of Call']).codes
    df['Location_Encoded'] = pd.Categorical(df['Location']).codes
    
    features = df[[
        'Number of Reports', 'Call Frequency', 'Duration (seconds)',
        'Confidence Score (%)', 'Time_Encoded', 'Location_Encoded'
    ]]
    
    # Scale numerical features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    features_scaled = pd.DataFrame(features_scaled, columns=features.columns)
    
    # Prepare text features
    vectorizer = TfidfVectorizer(
        max_features=50,
        min_df=3,
        max_df=0.7,
        stop_words='english'
    )
    text_features = vectorizer.fit_transform(df['Conversation Text'])
    text_features_dense = text_features.toarray()
    
    # Combine features
    X = np.hstack([features_scaled, text_features_dense])
    y = (df['Spam Classification'] == 'Spam').astype(float)  # Convert to float
    
    return X, y, vectorizer, scaler

def train_model(num_epochs=30, batch_size=32, learning_rate=0.001):
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load and prepare data
    X, y, vectorizer, scaler = load_and_prepare_data()
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create datasets and dataloaders
    train_dataset = SpamDataset(X_train, y_train)
    test_dataset = SpamDataset(X_test, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    input_dim = X.shape[1]
    model = SpamClassifier(input_dim)
    
    # Loss and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)  # Added L2 regularization
    
    # Training loop
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    print("Starting training...")
    print(f"Input dimension: {input_dim}")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    
    for epoch in range(num_epochs):
        model.train()
        epoch_train_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_X, batch_y in test_loader:
                outputs = model(batch_X)
                val_loss += criterion(outputs, batch_y).item()
                predicted = (outputs >= 0.5).float()
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        epoch_train_loss /= len(train_loader)
        val_loss /= len(test_loader)
        accuracy = 100 * correct / total
        
        train_losses.append(epoch_train_loss)
        val_losses.append(val_loss)
        
        # Print progress
        if (epoch + 1) % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], '
                  f'Train Loss: {epoch_train_loss:.4f}, '
                  f'Val Loss: {val_loss:.4f}, '
                  f'Accuracy: {accuracy:.2f}%')
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), 'spam_classifier_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after epoch {epoch+1}")
                break
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('training_history.png')
    plt.close()
    
    # Save preprocessing objects
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    joblib.dump(scaler, 'feature_scaler.joblib')
    
    print("\nTraining completed! Model and preprocessing objects saved.")

if __name__ == "__main__":
    train_model(num_epochs=30)