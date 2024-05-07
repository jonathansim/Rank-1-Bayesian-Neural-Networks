import json
import matplotlib.pyplot as plt
'''
This script loads a .json file containing the results from a model (e.g. a deterministic Wide ResNet), 
and then performs some results processing including plotting the validation accuracy, ECE and NLL over the various epochs. 
More functionality is to be added soon. 
'''

# Load in relevant file using JSON
filename = 'results_test.json'
with open(filename, 'r') as file:
    results = json.load(file)


# Extract relevant keys from the dictionaries (epoch, accuracy, nll, ece):
epochs = [d['epoch']+1 for d in results]
# epochs = [epoch+1 for epoch in epochs]
accuracies = [d['accuracy'] for d in results]
nll = [d['average_nll'] for d in results]
ece = [d['ece'] for d in results]

# Create a plot of accuracy over epochs 
plt.figure(figsize=(10, 5))
plt.plot(epochs, accuracies, marker='o', linestyle='-', color='b')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.show()