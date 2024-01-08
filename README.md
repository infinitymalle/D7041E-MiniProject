# Malkolm Lundkvist & Peggy Khialie

# D7041E-MiniProject
Mini project in course D7041E Applied Artificial Intelligence


# The article: 
Building Multilayer Perceptron Models in PyTorch
https://machinelearningmastery.com/building-multilayer-perceptron-models-in-pytorch/

# Installations
- Pytorch
- Dataset is imported in the code

# Link to Youtube video: 


# Result
Performances of models with different number of hidden layers for different cost/loss functions:

nn.CrossEntropyLoss()
- 0 Hidden layers: 97.5% 
- 2 Hidden layers: 96.95%
- 4 Hidden layers: 97.64%

nn.NLLLoss()
- 0 Hidden layers: 97.71% 
- 2 Hidden layers: 97.0%
- 4 Hidden layers: 97.29%

# Run the code
Remeber to comment out the LogSoftMax as explained in the video. Otherwise it is just to run the code. The expected result is documented in the code and should land araound 95-98 %. 
