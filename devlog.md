<h1 align="center">DevLog</h1>


## 2021-09-14

### Changes
- [x] Added loss functions dcfc6d8a1a8e7881aef93a98ed4c46093ed5cb97
  - [x] Binary Cross Entropy
  - [x] Categorical Cross Entropy 
- [x] Added fitting and loss calculations at `Model` class ce8d8cd052457c284f73c95c45a70b91b5bf1c27

### Challenges Faced
- [x] Loss can be `inf` due to `log(0)` in `CategoricalCrossEntropy` loss function. Added a small value(`EPSILON`) `1e-10` to avoid this.

## 2021-09-13
- [x] Neuron class added  907fae19b830c1244786a905cd80dbfd7ec0d0c7
- [x] Layers class added 907fae19b830c1244786a905cd80dbfd7ec0d0c7
- [x] Activation functions added fca8cbb332b3b33c45a27d5319abdc067330b0f2
- [x] Softmax Layer added ff1787c3ee27b426dc48a89b6bb57eef2d89d0db