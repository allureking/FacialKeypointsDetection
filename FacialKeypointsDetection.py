#%%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the datasets
path = '/Users/khh/Library/CloudStorage/OneDrive-Personal/STUDY/USF/Fall 2023/Advanced Machine Learning/FacialKeypointsDetection/'
training_data = pd.read_csv(path + 'training.csv')
test_data = pd.read_csv(path + 'test.csv')
sample_submission = pd.read_csv(path + 'SampleSubmission.csv')
id_lookup_table = pd.read_csv(path + 'IdLookupTable.csv')
# %%

# Function to plot keypoints on image
def plot_keypoints(image, keypoints):
    plt.imshow(image, cmap='gray')
    for i in range(0, len(keypoints), 2):
        if np.isnan(keypoints[i]) or np.isnan(keypoints[i+1]):
            continue
        plt.scatter(keypoints[i], keypoints[i+1], c='r', marker='x')
    plt.axis('off')

# Plotting the first 5 images with keypoints   
fig, axes = plt.subplots(1, 5, figsize=(20, 10))
for i, ax in enumerate(axes):
    plt.sca(ax)
    image = np.fromstring(training_data.iloc[i]['Image'], sep=' ').astype(np.uint8).reshape(96, 96)
    keypoints = training_data.iloc[i].drop('Image').values
    plot_keypoints(image, keypoints)

plt.title('First 5 Images with Keypoints')
plt.show()
# %%

# Data Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Handle missing values: Drop rows with missing keypoints
training_data_clean = training_data.dropna()

# Extract Image and Keypoints data
images = []
for idx, sample in training_data_clean.iterrows():
    image = np.fromstring(sample['Image'], sep=' ').astype(np.float32).reshape(96, 96)
    images.append(image)
images = np.array(images)

keypoints = training_data_clean.drop(columns=['Image']).to_numpy().astype(np.float32)

# Normalize images and keypoints
images = images / 255.0
# keypoints = (keypoints - 48) / 48  # Since 48 is approximately the center of the 96x96 image

# # Reshape to be compatible with neural network models
# images = np.expand_dims(images, axis=-1)

# Shuffle and Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, keypoints, test_size=0.2, random_state=42)

# Read in the data
# Treat it as a regression problem and use MSE as the loss function
# there are n keypoints and each keypoint has x and y coordinates
# so the output layer has 2n units
# For maximum prediction accuracy using a fully connected network,
# the number of layers should be

#%%
N_train = len(X_train)
i = np.random.randint(N_train)
img = X_train[i]
keypts = y_train[i]
x_coords = keypts[0::2]
y_coords = keypts[1::2]
plt.figure()
plt.imshow(img, cmap='gray')    
plt.scatter(x_coords, y_coords, c='r', marker='.')
plt.show()


# %%
import torch

class FacialKeypointsDataset(torch.utils.data.Dataset):
    
    def __init__(self, X, y):
        self.X = X
        self.y = y
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        
        img = self.X[i]
        keypts = self.y[i]
        
        img = np.reshape(img, (1, 96, 96))
        img = torch.tensor(img, dtype=torch.float32)
        keypts = torch.tensor(keypts, dtype=torch.float32)

        return img, keypts
    
class FacialKeypointsTestDataset(torch.utils.data.Dataset):
    
    def __init__(self, X):
        self.X = X
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, i):
        
        img = self.X[i]
        img = np.reshape(img, (1, 96, 96))
        img = torch.tensor(img, dtype=torch.float32)

        return img
    
# %%
dataset_train = FacialKeypointsDataset(X_train, y_train)
dataset_val = FacialKeypointsDataset(X_val, y_val)


# %%
N_train = len(dataset_train)
i = np.random.randint(N_train)
img, keypts = dataset_train.__getitem__(i)
x_coords = keypts[0::2]
y_coords = keypts[1::2]
plt.figure()
plt.imshow(img.squeeze(), cmap = 'gray')
plt.scatter(x_coords, y_coords, c='r', marker='.')

# %%
dataloader_train = torch.utils.data.DataLoader(dataset_train, batch_size=16, shuffle=True)
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=16, shuffle=False) 

#%%
x_batch, y_batch = next(iter(dataloader_train))

# %%
# Define the neural network model
import torch.nn as nn
import torch.nn.functional as F

class FacialKeypointsNet(nn.Module):
    def __init__(self):
        super(FacialKeypointsNet, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, 5)  # 1 input channel, 32 output channels, 5x5 kernel
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)
        
        # Fully connected layers
        self.fc1 = nn.Linear(128*10*10, 256)  # Flattened dimensions: 128x10x10
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 30)  # 30 outputs (15 keypoints * 2 (x, y))
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        
        x = x.view(x.size(0), -1)  # Flatten
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x

# Instantiate the model
model = FacialKeypointsNet()

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#%%
import torch.optim as optim

# Loss Function & Optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training Loop
num_epochs = 50

for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    for images, keypoints in dataloader_train:
        # Zero the parameter gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, keypoints)
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Print loss statistics
    train_loss = train_loss/len(dataloader_train)
    print(f'Epoch: {epoch+1}, Training Loss: {train_loss:.4f}')

#%%

# Validation:
model.eval()
with torch.no_grad():
    val_loss = 0.0
    for images, keypoints in dataloader_val:
        images, keypoints = images.to(device), keypoints.to(device)
        outputs = model(images)
        loss = criterion(outputs, keypoints)
        val_loss += loss.item()
    val_loss = (val_loss / len(dataloader_val)) * 100
    print(f'Validation Loss: {val_loss:.2f}%')
