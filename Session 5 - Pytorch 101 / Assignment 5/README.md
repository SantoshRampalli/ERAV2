# Assignment 5

## Usage
- Run the notebook `S5.ipynb` to train & test the model
- Model Architecture in `model.py`
- Helper functions in `utils.py`
---

## Solution Details
### Modules
- `model.py`: convolutional neural network classes
- `utils.py`: helper functions for the model training
  - `train`: train model on the training set
  - `test`: test using the testing set

### Model Summary
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 32, 26, 26]             320
            Conv2d-2           [-1, 64, 24, 24]          18,496
            Conv2d-3          [-1, 128, 10, 10]          73,856
            Conv2d-4            [-1, 256, 8, 8]         295,168
            Linear-5                   [-1, 50]         204,850
            Linear-6                   [-1, 10]             510
================================================================
Total params: 593,200
Trainable params: 593,200
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.00
Forward/backward pass size (MB): 0.67
Params size (MB): 2.26
Estimated Total Size (MB): 2.94
----------------------------------------------------------------
```
