# PSO-tuning-on-Federated-learning
Using PSO algorithm to tune hyper-parameters in Federated learning environment

Experiments on MNIST and CIFAR10 (both IID and non-IID) can be produced

The settings can be changed

## Run

Federated learning tuning process using PSO algorithm is produced by:
> python [main_pso.py](main_fed.py)

In comparsion, the FL tuning process with GA is produced by:
> python [main_ga.py](main_ga.py)

For example:
> python main_pso.py --dataset mnist --iid --num_channels 1 --model cnn --epochs 50 --gpu 0  

`--all_clients` for averaging over all client models

NB: for CIFAR-10, `num_channels` must be 3.
