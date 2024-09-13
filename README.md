<div align="center">
    <h1>Amber</h1>
    <p> Amber is a custom made Neural Network library written in Python. I am build this libray to learn more about Neural Networks and how they work. Maybe in the future I will use this library to build some cool projects.
    </p>
</div>

## Requirements 📦
- Python 3.6 or higher
- Numpy == 2.1.1

## Components Implemented 🧩
- [x] Models
- [x] Layers
  - [x] Input
  - [x] Dense 
  - [x] Softmax
- [x] Activation Functions
  - [x] ReLU
  - [x] Sigmoid

## Usage 🚀

```python
from amber.models import Models
from amber.layers import Dense, Input, Softmax

model = Models([
    Input(2),
    Dense(3, activation='relu'),
    Dense(2, activation='sigmoid'),
    Softmax()
])

model.compile()

x = np.array([1.3, 2.7])

y = model.forward(x)
```
## Future Plans 🚀
- [ ] Implement more layers
- [ ] Implement more activation functions
- [ ] Implement Backpropagation
- [ ] Implement Optimizers
- [ ] Implement Loss Functions

## Directory Structure 📁
```bash
├── amber
│   ├── layers.py
│   ├── models.py
│   ├── activations.py
│   ├── nurons.py
│   ├── utils.py
```

## Support Me 🙏
If you like this project, consider supporting me. You can do this by:
- Starring this repository
- Sharing this repository with your friends
- Following me on [Twitter](https://x.com/iamBijoyKar)
