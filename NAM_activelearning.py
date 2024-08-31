import numpy as np
import torch
from torch import nn
import csv
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

device = "cuda" if torch.cuda.is_available() else "cpu"  # sets up running environment

# DATA GENERATION

def csv_to_matrix(file_path):
    matrix = []
    with open(file_path, mode='r', newline='') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            matrix.append([float(cell) for cell in row])
    return matrix

# Example usage:
file_path = '/Users/bkolicic/Documents/Cleaned Data.csv'  # Phishing Data
matrix = csv_to_matrix(file_path)



# Load Labels
# Specify the path to your labels file
csv_file_path = '/Users/bkolicic/Documents/Intrusion Labels.csv'

# Initialize an empty dictionary
labels_hashmap = {}

# Open and read the CSV file
with open(csv_file_path, mode='r', newline='') as file:
    csv_reader = csv.reader(file)
    
    # Iterate through the rows in the CSV
    for row in csv_reader:
        # Assuming the labels are in the first row
        labels = row
        break

# Populate the dictionary with the labels and their index as the value
for index, label in enumerate(labels):
    labels_hashmap[index + 1] = label
    
    
    
    
matrix = np.array(matrix)

# Set random seed for reproducibility



# Generate sample data
n_samples = len(matrix)  # Number of samples
n_features = (len(matrix[0]) - 1)  # Number of features in the context
train_proportion = 0.8  # Proportion of training data in the dataset

X = matrix[:, :n_features]
L = matrix[:, n_features]

# create train/test data split
train_split = int(train_proportion * len(X))  # 80% of data used for training set, 20% for testing
X_train, L_train = X[:train_split], L[:train_split]
X_test, L_test = X[train_split:], L[train_split:]

# Convert numpy arrays to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
L_train = torch.tensor(L_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
L_test = torch.tensor(L_test, dtype=torch.float32).to(device)

# MODEL CREATION

# Define the Neural Additive Model
class NAM(nn.Module):
    def __init__(self, n_features):
        super(NAM, self).__init__()
        self.feature_nets = nn.ModuleList([self._create_feature_net() for _ in range(n_features)])
        self.output_layer = nn.Linear(n_features, 1)

    def _create_feature_net(self):
        return nn.Sequential(
            nn.Linear(1, 10),  # Adjust the number of neurons as needed
            nn.ReLU(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        feature_outputs = [net(x[:, i:i+1]) for i, net in enumerate(self.feature_nets)]
        combined_output = torch.cat(feature_outputs, dim=1)
        output = self.output_layer(combined_output)
        return output, feature_outputs

model = NAM(n_features).to(device)

# Create a loss function
loss_fn = nn.BCEWithLogitsLoss()  # BCEWithLogitsLoss = sigmoid built-in

# Create an optimizer
optimizer = torch.optim.SGD(params=model.parameters(), lr=0.005)

# Calculate accuracy (a classification metric)
def accuracy_fn(L_true, L_pred):
    correct = torch.eq(L_true, L_pred).sum().item()  # torch.eq() calculates where two tensors are equal
    acc = (correct / len(L_pred)) * 100
    return acc

# TRAINING AND TESTING

# Put data to target device
X_train, L_train = X_train.to(device), L_train.to(device)
X_test, L_test = X_test.to(device), L_test.to(device)

epochs = 500  # Set the number of epochs
active_learning_on = True  # Will we use active learning
window_proportion = 0.1  # Size of window compared to size of train set
buffer_size = 0.1  # Buffer split (how much in "recents" set)
batch_check_threshold = 0.1  # What percentage of the batch can we check
max_batch_size = 100  # What is the maximum expected batch size

window_size = int(len(X_train) * window_proportion)
buffer_split = int(buffer_size * window_size)  # How much of buffer will be chosen randomly from the data

accuracy_count = 0  # For accuracy metric
TP = 0
FP = 0
TN = 0
FN = 0

predicted_probs = torch.empty((0), dtype=torch.float32)  # To store predicted probabilities for ROC curve
true_labels = torch.empty((0), dtype=torch.float32)  # To store true labels for ROC curve

# Initialize feature_output_test as a list of empty tensors
feature_outputs_test = [torch.empty(0, 1).to(device) for _ in range(n_features)]

min_mean_counter = np.zeros((1, n_features)) # Matrix for storing which characteristic comes up most
max_variance_counter = np.zeros((1, n_features)) # Matrix for storing which characteristic has most variance

test_pred = torch.empty(0, dtype=torch.float32)

while len(X_test) != 0:
    # Take new batch
    batch_size = np.random.randint(int(max_batch_size / 2), max_batch_size + 1)  # Choose random batch size
    if batch_size > len(X_test): batch_size = len(X_test)
    batch_check_no = int(batch_check_threshold * batch_size)  # How many samples can we check are correct
    batch_variances = np.zeros((batch_size, 1))  # Vector for saving batch and variances, for active learning

    # Choose recent contexts and random contexts for buffer
    X_window = X_train[-buffer_split:, :].cpu().numpy()  # last elements of X_train
    random_indices = np.random.randint(0, int(len(X_train)) - buffer_split, size=buffer_split)
    X_window = np.vstack([X_window, X_train[random_indices].cpu().numpy()])
    L_window = np.concatenate([L_train[-buffer_split:].cpu().numpy(), L_train[random_indices].cpu().numpy()])

    # Convert back to tensors and move to device
    X_window = torch.tensor(X_window, dtype=torch.float32).to(device)
    L_window = torch.tensor(L_window, dtype=torch.float32).squeeze().to(device)

    # Build training and evaluation loop each batch
    for epoch in range(epochs):
        ### Training
        model.train()
        # 1. Forward pass (model outputs raw logits)
        L_logits, _ = model(X_window)  # squeeze to remove extra `1` dimensions
        L_pred = torch.round(torch.sigmoid(L_logits))  # turn logits -> pred probs -> pred labels

        # 2. Calculate loss/accuracy
        loss = loss_fn(L_logits, L_window.unsqueeze(1))  # Using nn.BCEWithLogitsLoss works with raw logits

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backwards
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        ### Testing
        model.eval()
        with torch.inference_mode():
            # 1. Forward pass
            test_logits, batch_feature_outputs = model(X_test[:batch_size])
            test_logits = test_logits.squeeze()
            batch_pred = torch.round(torch.sigmoid(test_logits))
            for i in range(n_features):
                batch_feature_outputs[i] = torch.sigmoid(batch_feature_outputs[i])
            # 2. Calculate loss/accuracy
            test_loss = loss_fn(test_logits, L_test[:batch_size])
            test_acc = accuracy_fn(L_true=L_test[:batch_size], L_pred=batch_pred)

    for i in range(n_features):
        feature_outputs_test[i] = torch.cat((feature_outputs_test[i], batch_feature_outputs[i]), dim=0)

    for i in range(len(batch_pred)):  # Metric Calculation for each item in batch
        if batch_pred[i] == L_test[i]: accuracy_count += 1
            
        if batch_pred[i] == 0 and L_test[i] == 0:
                TP += 1
        elif batch_pred[i] == 0 and L_test[i] == 1:
                FP += 1
        elif batch_pred[i] == 1 and L_test[i] == 0:
                FN += 1
        elif batch_pred[i] == 1 and L_test[i] == 1:
                TN += 1

        batch_variances[i] = torch.sigmoid(test_logits[i].squeeze()) * (1 - torch.sigmoid(test_logits[i].squeeze()))  # Batch Variances for active learning
   
    print(batch_pred)
    
    if active_learning_on:
        X_train = torch.cat([X_train, X_test[:batch_size]], dim=0)  # add batch data to training data
        indices_to_check = np.argsort(batch_variances.squeeze())[-batch_check_no:]  # Find indices of largest variance
        batch_pred_clone = batch_pred.clone()  # Create a clone of test_pred for in-place operations, as cant perform inplace operation on device
        batch_pred_clone[indices_to_check] = L_test[indices_to_check]  # Correct values with highest variance
        L_train = torch.cat([L_train, batch_pred_clone], dim=0)  # add corrected batch labels to training data
    else:
        X_train = torch.cat([X_train, X_test[:batch_size]], dim=0)  # add batch data to training data
        L_train = torch.cat([L_train, L_test[:batch_size]], dim=0)  # add batch labels to training data


    predicted_probs = torch.cat((predicted_probs,(torch.sigmoid(test_logits))),dim = 0)
    true_labels = torch.cat((true_labels,(L_test[:batch_size])), dim=0)

    # Update the test set
    X_test = X_test[batch_size:]
    L_test = L_test[batch_size:]
    test_pred = torch.cat((test_pred, batch_pred), dim=0)
    
    
    
# METRICS


for i in range(len(feature_outputs_test[0])):
    for j in range(len(feature_outputs_test)):
        # Use .item() to get the scalar value for comparison
        if j == 0 or feature_outputs_test[j][i].mean().item() <= feature_outputs_test[index_of_min_mean][i].mean().item():
            index_of_min_mean = j
        if j == 0 or (feature_outputs_test[j][i].mean().item())*(1-feature_outputs_test[j][i].mean().item()) >= (feature_outputs_test[index_of_max_variance][i].mean().item())*(1-feature_outputs_test[index_of_max_variance][i].mean().item()):
            index_of_max_variance = j
    if test_pred[i].item() == 0:  # Use .item() to get the scalar value from the tensor
        min_mean_counter[0, index_of_min_mean] += 1
    max_variance_counter[0,index_of_max_variance] += 1
    



print(min_mean_counter)
print(max_variance_counter)
mean_counter_sum = np.sum(min_mean_counter)
variance_counter_sum = np.sum(max_variance_counter)

for i in range(n_features):
    if min_mean_counter[0][i] > 0:
        print("Characteristic ", labels_hashmap[i+1], " contributes the most to a decision for", np.round(min_mean_counter[0][i]/mean_counter_sum * 100, 1), "% of the time")
        
for i in range(n_features):
    if max_variance_counter[0][i] > 0:
        print("Characteristic ", labels_hashmap[i+1], " has highest variance", np.round(max_variance_counter[0][i]/variance_counter_sum * 100, 1), "% of the time")




true_labels = true_labels.numpy() # Convert to Tensor
predicted_probs = predicted_probs.numpy()

    
    
print("accuracy = ", accuracy_count / (n_samples - int(train_proportion * n_samples)))
print("TP% = ", TP / (n_samples - int(train_proportion * n_samples)))
print("FP% = ", FP / (n_samples - int(train_proportion * n_samples)))
print("FN% = ", FN / (n_samples - int(train_proportion * n_samples)))
print("TN% = ", TN / (n_samples - int(train_proportion * n_samples)))
print("F1 = ", TP / (TP + 0.5 * (FP + FN)))


# Calculate ROC curve and ROC AUC score
fpr, tpr, _ = roc_curve(true_labels,1-np.array( predicted_probs), pos_label=0) # 1-P(1) = P(0) calculate correct probilities
roc_auc = roc_auc_score(true_labels,predicted_probs)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
#plt.show()

print("ROC AUC Score: ", roc_auc)


