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
rep 1, Loss: 0.3883, Test error: 7.60%
rep 2, Loss: 0.4158, Test error: 7.40%
rep 3, Loss: 0.3051, Test error: 6.80%
rep 4, Loss: 0.4440, Test error: 7.80%
rep 5, Loss: 0.3056, Test error: 5.80%
rep 6, Loss: 0.3374, Test error: 6.40%
rep 7, Loss: 0.3037, Test error: 5.80%
rep 8, Loss: 0.3707, Test error: 6.20%
rep 9, Loss: 0.3816, Test error: 6.00%
rep 10, Loss: 0.3055, Test error: 5.60%
Final result, loss: 0.36 ± 0.05, error: 6.54% ± 0.81%
```
