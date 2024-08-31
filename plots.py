import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel('/Users/bkolicic/OneDrive - The Alan Turing Institute/Experiments.xlsx', sheet_name='FINAL')

# ROC Curves

NN_fpr = df['NN_FPR']
NN_tpr = df['NN_TPR']
NAM_fpr = df['NAM_FPR']
NAM_tpr = df['NAM_TPR']
GP_fpr = df['GP_FPR']
GP_tpr = df['GP_TPR']
GPNAM_fpr = df['GPNAM_FPR']
GPNAM_tpr = df['GPNAM_TPR']

# Window Proportion

WINDOWPROP = df['WINDOWPROPORTION']
NN_WINDOW = df['NN_WINDOW']
GP_WINDOW = df['GP_WINDOW']
NAM_WINDOW = df['NAM_WINDOW']
GPNAM_WINDOW = df['GPNAM_WINDOW']
NN_WINDOWVAR = df['NN_WINDOWVAR']
GP_WINDOWVAR = df['GP_WINDOWVAR']
NAM_WINDOWVAR = df['NAM_WINDOWVAR']
GPNAM_WINDOWVAR = df['GPNAM_WINDOWVAR']

# Batch Proportion 

BATCHPROP = df['BATCHPROPORTION']
NN_BATCH = df['NN_BATCH']
GP_BATCH = df['GP_BATCH']
NAM_BATCH = df['NAM_BATCH']
GPNAM_BATCH = df['GPNAM_BATCH']
NN_BATCHVAR = df['NN_BATCHVAR']
GP_BATCHVAR = df['GP_BATCHVAR']
NAM_BATCHVAR = df['NAM_BATCHVAR']
GPNAM_BATCHVAR = df['GPNAM_BATCHVAR']


# Feature Contributions

CONTRIBUTIONFEATURES = df['CONTRIBUTIONFEATURES']
GPNAM_CONTRIBUTOR = df['GPNAM_CONTRIBUTOR']
NAM_CONTRIBUTOR = df['NAM_CONTRIBUTOR']


# Variance Contributions

VARIANCEFEATURES = df['VARIANCEFEATURES']
GPNAM_VARIANCE = df['GPNAM_VARIANCE']
NAM_VARIANCE = df['NAM_VARIANCE']


# ROC Plot

plt.plot(NN_fpr, NN_tpr, label='NN', color='blue')
plt.plot(GP_fpr, GP_tpr, label='GP', color='green')
plt.plot(NAM_fpr, NAM_tpr, label='NAM', color='red')
plt.plot(GPNAM_fpr, GPNAM_tpr, label='GPNAM', color='purple')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.show()

# Window Proportion


plt.plot(WINDOWPROP, NN_WINDOW, label='NN', color='blue')
plt.plot(WINDOWPROP, GP_WINDOW, label='GP', color='green')
plt.plot(WINDOWPROP, NAM_WINDOW, label='NAM', color='red')
plt.plot(WINDOWPROP, GPNAM_WINDOW, label='GPNAM', color='purple')
plt.errorbar(WINDOWPROP, NN_WINDOW, yerr=NN_WINDOWVAR/np.sqrt(5), fmt='x', capsize=5, color='blue')
plt.errorbar(WINDOWPROP, GP_WINDOW, yerr=GP_WINDOWVAR/np.sqrt(5), fmt='x', capsize=5, color='green')
plt.errorbar(WINDOWPROP, NAM_WINDOW, yerr=NAM_WINDOWVAR/np.sqrt(5), fmt='x', capsize=5, color='red')
plt.errorbar(WINDOWPROP, GPNAM_WINDOW, yerr=GPNAM_WINDOWVAR/np.sqrt(5), fmt='x', capsize=5, color='purple')
plt.xticks(np.arange(min(WINDOWPROP), max(WINDOWPROP) + 0.1, 0.1))
plt.xlabel('Window Proportion')
plt.ylabel('F1 Score')
plt.legend()
plt.show()

# Batch Proportion

plt.plot(BATCHPROP, NN_BATCH, label='NN', color='blue')
plt.plot(BATCHPROP, GP_BATCH, label='GP', color='green')
plt.plot(BATCHPROP, NAM_BATCH, label='NAM', color='red')
plt.plot(BATCHPROP, GPNAM_BATCH, label='GPNAM', color='purple')
plt.errorbar(BATCHPROP, NN_BATCH, yerr=NN_BATCHVAR/np.sqrt(5), fmt='x', capsize=5, color='blue')
plt.errorbar(BATCHPROP, GP_BATCH, yerr=GP_BATCHVAR/np.sqrt(5), fmt='x', capsize=5, color='green')
plt.errorbar(BATCHPROP, NAM_BATCH, yerr=NAM_BATCHVAR/np.sqrt(5), fmt='x', capsize=5, color='red')
plt.errorbar(BATCHPROP, GPNAM_BATCH, yerr=GPNAM_BATCHVAR/np.sqrt(5), fmt='x', capsize=5, color='purple')
plt.xticks(np.arange(min(BATCHPROP), max(BATCHPROP) + 0.1, 0.1))
plt.xlabel('Batch Proportion')
plt.ylabel('F1 Score')
plt.legend()
plt.show()

# Contributions

CONTRIBUTIONFEATURES = CONTRIBUTIONFEATURES.dropna()
CONTRIBUTIONFEATURES = CONTRIBUTIONFEATURES.astype(int)
GPNAM_CONTRIBUTOR = GPNAM_CONTRIBUTOR.dropna()
NAM_CONTRIBUTOR = NAM_CONTRIBUTOR.dropna()

bar_width = 0.35  # Width of each bar
index = np.arange(len(CONTRIBUTIONFEATURES))  # The x locations for the groups
# Create the bar plot
plt.bar(index - bar_width / 2, GPNAM_CONTRIBUTOR, bar_width, label='GPNAM', color='purple')
plt.bar(index + bar_width / 2, NAM_CONTRIBUTOR, bar_width, label='NAM', color='red')
# Add labels, title, and legend
plt.xlabel('Feature')
plt.ylabel('%')
plt.xticks(index, CONTRIBUTIONFEATURES)  # Set the x-ticks to be the feature names
plt.legend()
plt.show()


# Variances 

VARIANCEFEATURES = VARIANCEFEATURES.dropna()
VARIANCEFEATURES=VARIANCEFEATURES.astype(int)
GPNAM_VARIANCE = GPNAM_VARIANCE.dropna()
NAM_VARIANCE = NAM_VARIANCE.dropna()

bar_width = 0.35  # Width of each bar
index = np.arange(len(VARIANCEFEATURES))  # The x locations for the groups
# Create the bar plot
plt.bar(index - bar_width / 2, GPNAM_VARIANCE, bar_width, label='GPNAM', color='purple')
plt.bar(index + bar_width / 2, NAM_VARIANCE, bar_width, label='NAM', color='red')
# Add labels, title, and legend
plt.xlabel('Feature')
plt.ylabel('%')
plt.xticks(index, VARIANCEFEATURES)  # Set the x-ticks to be the feature names
plt.legend()
plt.show()



