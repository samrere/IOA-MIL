# IOA-MIL
dataset used are made from MNIST and it's the same as the paper "attention based deep multiple instance learning".
## How to use
```
$ python main.py --model A
choosing the model from "Attention based deep multiple instance learning"
repeating 10 times...
rep 1, Loss: 0.4565, Test error: 8.60%
rep 2, Loss: 0.5648, Test error: 8.40%
rep 3, Loss: 0.4386, Test error: 7.20%
rep 4, Loss: 0.2610, Test error: 7.60%
rep 5, Loss: 0.4498, Test error: 8.00%
rep 6, Loss: 0.4120, Test error: 7.40%
rep 7, Loss: 0.5943, Test error: 9.00%
rep 8, Loss: 0.4024, Test error: 9.20%
rep 9, Loss: 0.6344, Test error: 9.60%
rep 10, Loss: 0.3215, Test error: 6.80%
Final result, loss: 0.45 ± 0.12, error: 8.18% ± 0.93%

$ python main.py --model B
choosing my model with instance loss
repeating 10 times...
rep 1, Loss: 0.3658, Test error: 7.80%
rep 2, Loss: 0.4105, Test error: 7.80%
rep 3, Loss: 0.3828, Test error: 6.60%
rep 4, Loss: 0.3722, Test error: 7.20%
rep 5, Loss: 0.3958, Test error: 6.60%
rep 6, Loss: 0.4002, Test error: 6.20%
rep 7, Loss: 0.4805, Test error: 8.20%
rep 8, Loss: 0.3508, Test error: 6.80%
rep 9, Loss: 0.3119, Test error: 5.60%
rep 10, Loss: 0.3367, Test error: 5.80%
Final result, loss: 0.38 ± 0.05, error: 6.86% ± 0.88%
```
