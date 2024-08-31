import GPy
import numpy as np
import matplotlib.pyplot as plt
import csv
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score


def csv_to_matrix(file_path):
    matrix = []
    with open(file_path, mode='r', newline='') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:
            matrix.append([float(cell) for cell in row])
    return matrix

file_path = 'datafile.csv'
matrix = csv_to_matrix(file_path)

matrix = np.array(matrix)
np.random.shuffle(matrix)

n_samples = len(matrix)  # Number of samples
n_features = len(matrix[0]) - 1  # Number of features in the context

# Model Parameters

train_proportion = 0.8  # Proportion of training data in the dataset
sigma = 1
l = 1
active_learning_on = False  # Will we use active learning   
window_proportion = 0.1  # Size of window compared to size of train set
buffer_size = 0.1  # Buffer split (how much in "recents" set)
batch_check_threshold = 0.1  # What percentage of the batch can we check
max_batch_size = 100  # What is the maximum expected batch size

X = matrix[:, :n_features]  # Ensure X includes all feature columns
L = matrix[:, n_features]
L = L.reshape(-1, 1)

# Create train/test data split
train_split = int(train_proportion * len(X))  # 80% of data used for training set, 20% for testing
X_train, L_train = X[:train_split], L[:train_split]
X_test, L_test = X[train_split:], L[train_split:]

# MODEL CREATION 

# Create separate additive kernels for each dimension
kernel = GPy.kern.RBF(input_dim=n_features, variance=sigma, lengthscale=l)

window_size = int(len(X_train) * window_proportion)
buffer_split = int(buffer_size * window_size)  # How much of buffer will be chosen randomly from the data

# Choose recent contexts and random contexts for buffer
X_window = X_train[-buffer_split:, :]  # last elements of X_train
random_indices = np.random.randint(0, int(len(X_train)) - buffer_split, size=buffer_split)
X_window = np.vstack([X_window, X_train[random_indices]])
L_window = np.vstack([L_train[-buffer_split:], L_train[random_indices]])

model = GPy.models.GPRegression(X_window, L_window, kernel)
model.optimize(messages=True)

# Initialize logistic regression model
logistic_model = LogisticRegression()

# Train logistic regression model on the initial training data
initial_means = np.zeros((len(X_window), 1))
for i in range(len(X_window)):
    mean_contribution, _ = model.predict(X_window[i].reshape(1, -1))
    initial_means[i] = mean_contribution

logistic_model.fit(initial_means, L_window.ravel())

# Updating

accuracy_count = 0  # For accuracy metric
TP = 0
FP = 0
TN = 0
FN = 0

predicted_probs = []  # To store predicted probabilities for ROC curve
true_labels = []  # To store true labels for ROC curve

while len(X_test) != 0:  # While testing set is not empty

    batch_size = np.random.randint(int(max_batch_size / 2), max_batch_size + 1)  # Choose random batch size
    if batch_size > len(X_test): batch_size = len(X_test)
    batch_check_no = int(batch_check_threshold * batch_size)  # How many samples can we check are correct
    batch_contributions = np.zeros((batch_size, 2))  # Matrix for saving batch means and variances
    # Empty matrices for holding context info
    batch_X = np.zeros((batch_size, n_features))
    batch_L_pred = np.zeros((batch_size, 1))
    batch_L_true = np.zeros((batch_size, 1))

    for i in range(batch_size):  # for each context in testing set

        current_context_row = X_test[i].reshape(1, -1)  # Take current context

        mean_contribution, variance_contribution = model.predict(current_context_row)  # Find individual contributions of mean and variance for each feature

        # Predict using logistic regression model
        L_pred_prob = logistic_model.predict_proba(mean_contribution)[:, 0]
        L_pred = (L_pred_prob <= 0.5).astype(int)

        predicted_probs.append(L_pred_prob[0])
        true_labels.append(L_test[i][0])

        if L_pred == L_test[i]:
            accuracy_count += 1  # accuracy counter

        if L_pred == 0 and L_test[i] == 0:
            TP += 1
        elif L_pred == 0 and L_test[i] == 1:
            FP += 1
        elif L_pred == 1 and L_test[i] == 0:
            FN += 1
        elif L_pred == 1 and L_test[i] == 1:
            TN += 1

        print("Predicted: ", L_pred, "Actual: ", L_test[i], "Variance: ", variance_contribution)

        batch_contributions[i, 0] = mean_contribution
        batch_contributions[i, 1] = variance_contribution

        batch_X[i] = current_context_row
        batch_L_pred[i] = L_pred
        batch_L_true[i] = L_test[i]

    if active_learning_on:  # If we allow active learning

        X_train = np.vstack([X_train, batch_X])  # add context row to training data
        indices_to_check = np.argsort(batch_contributions[:, 1])[-batch_check_no:]  # Find indices of largest variance
        batch_L_pred[indices_to_check] = batch_L_true[indices_to_check]  # Correct values with highest variance

        L_train = np.vstack([L_train, batch_L_pred])  # add mean prediction to training data

        # Choose recent contexts and random contexts for buffer
        X_window = X_train[-buffer_split:, :]  # last elements of X_train
        random_indices = np.random.randint(0, int(len(X_train)) - buffer_split, size=buffer_split)
        X_window = np.vstack([X_window, X_train[random_indices]])
        L_window = np.vstack([L_train[-buffer_split:], L_train[random_indices]])

        initial_means = np.zeros((len(X_window), 1))
        for i in range(len(X_window)):
            mean_contribution, _ = model.predict(X_window[i].reshape(1, -1))
            initial_means[i] = mean_contribution

        # Update logistic regression model with new data
        logistic_model.fit(initial_means, L_window.ravel())

        model.set_XY(X_window, L_window)  # update model  
        model.optimize(messages=False)  # reoptimize model

        X_test = X_test[batch_size:]
        L_test = L_test[batch_size:]

    else:
        X_train = np.vstack([X_train, batch_X])  # add context row to training data
        L_train = np.vstack([L_train, batch_L_true])  # add mean prediction to training data

        # Choose recent contexts and random contexts for buffer
        X_window = X_train[-buffer_split:, :]  # last elements of X_train
        random_indices = np.random.randint(0, int(len(X_train)) - buffer_split, size=buffer_split)
        X_window = np.vstack([X_window, X_train[random_indices]])
        L_window = np.vstack([L_train[-buffer_split:], L_train[random_indices]])

        initial_means = np.zeros((len(X_window), 1))
        for i in range(len(X_window)):
            mean_contribution, _ = model.predict(X_window[i].reshape(1, -1))
            initial_means[i] = mean_contribution

        # Update logistic regression model with new data
        logistic_model.fit(initial_means, L_window.ravel())

        model.set_XY(X_window, L_window)  # update model  
        model.optimize(messages=False)  # reoptimize model

        X_test = X_test[batch_size:]
        L_test = L_test[batch_size:]

print("accuracy = ", accuracy_count / (n_samples - int(train_proportion * n_samples)))
print("TP% = ", TP / (n_samples - int(train_proportion * n_samples)))
print("FP% = ", FP / (n_samples - int(train_proportion * n_samples)))
print("FN% = ", FN / (n_samples - int(train_proportion * n_samples)))
print("TN% = ", TN / (n_samples - int(train_proportion * n_samples)))
print("F1 = ", TP / (TP + 0.5 * (FP + FN)))

# Calculate ROC curve and ROC AUC score
fpr, tpr, _ = roc_curve(true_labels, predicted_probs, pos_label=0)
roc_auc = roc_auc_score(true_labels,1-np.array(predicted_probs))

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
plt.show()

print("ROC AUC Score: ", roc_auc)

