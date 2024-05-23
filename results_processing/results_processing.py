import json
import matplotlib.pyplot as plt
'''
This script loads a .json file containing the results from a model (e.g. a deterministic Wide ResNet), 
and then performs some results processing including plotting the validation accuracy, ECE and NLL over the various epochs. 
More functionality is to be added soon. 
'''

# Load in relevant file using JSON
filename = 'det_val_results_05-21-H14.json'
with open(filename, 'r') as file:
    results = json.load(file)


# Extract relevant keys from the dictionaries (epoch, accuracy, nll, ece):
epochs = [d['epoch']+1 for d in results]
# epochs = [epoch+1 for epoch in epochs]
accuracies = [d['accuracy'] for d in results]
nll = [d['average_nll'] for d in results]
ece = [d['ece'] for d in results]

# Create a plots of metrics over epochs 
plt.figure(figsize=(20, 7))


# Subplot for Accuracy
plt.subplot(1, 3, 1)  # 1 row, 3 columns, 1st subplot
plt.plot(epochs, accuracies, marker='o', linestyle='-', color='b')
plt.title('Model Validation Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)

# Subplot for NLL
plt.subplot(1, 3, 2)  # 1 row, 3 columns, 2nd subplot
plt.plot(epochs, nll, marker='o', linestyle='-', color='r')
plt.title('Validation Negative Log Likelihood Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('NLL')
plt.grid(True)

# Subplot for ECE
plt.subplot(1, 3, 3)  # 1 row, 3 columns, 3rd subplot
plt.plot(epochs, ece, marker='o', linestyle='-', color='g')
plt.title('Validation Expected Calibration Error Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('ECE')
plt.grid(True)

# Show plot
plt.tight_layout()
plt.show()