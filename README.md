# On training Physics-Informed Neural Networks for Oscillating Problems

This repositorty contains the code to reproduce the results of the **ICLR 2024 Workshop on AI4DifferentialEquations In Science** paper "On training Physics-Informed Neural Networks for Oscillating Problems" (https://openreview.net/forum?id=YZ6wcrQE62): 

>**Abstract**:
Physics-Informed Neural Networks (PINNs) offer an efficient approach to solving partial differential equations (PDEs). In theory, they can provide the solution to a PDE at an arbitrary point for the computational cost of a single forward pass of a neural network. However, PINNs often pose challenges during training, necessitating complex hyperparameter tuning, particularly for PDEs with oscillating solutions. In this paper, we propose a PINN training scheme for PDEs with oscillating solutions. We analyze the impact of sinusoidal activation functions as model prior and incorporate self-adaptive weights into the training process. Our experiments utilize the double mass-spring-damper system to examine shortcomings in training PINNs. Our results show that strong model priors, such as sinusoidal activation functions, are immensely beneficial and, combined with self-adaptive training, significantly improve performance and convergence of PINNs.

# Sources

The implementation of the PINNs is based on the "Understanding the Difficulty of Training Physics-Informed Neural Networks on Dynamical Systems repository" (https://github.com/frohrhofer/PINN_TF2).

The implementation of the self-adaptive physics-informed neural networks [[1]](#1) is based on the original implementation (https://github.com/levimcclenny/SA-PINNs).

To obtain the results for the wave equation mentioned in the appendix, we adapted the source code from [[2]](#2) by adding self-adaptive weights and the different configurations of activation functions needed for our experiments. The original code can be found at (https://github.com/ShotaDeguchi/PINN_TF2)

### References
<a id="1">[1]</a> McClenny, L., & Braga-Neto, U. (2020). Self-adaptive physics-informed neural networks using a soft attention mechanism. arXiv preprint arXiv:2009.04544.

<a id="2">[2]</a> Deguchi, S., Shibata, Y., & Asai, M. (2021). Unknown parameter estimation using physics-informed neural networks with noised observation data. Journal of Japan Society of Civil Engineers, Ser. A2 (Applied Mechanics (AM)), 77(2), I_35-I_45.

# Usage
The code to solve the differential equation for the double mass-spring-damper system can be found in `two_mass_pinn_va.py` and `two_mass_pinn_sa.py` for the vanilla and the self-adaptive PINN respectively.
To change the activation functions set in `main()`:
```
# for tanh activation functions in all hidden layers
act_func_str = "tanh"

# for sine activation functions in all hidden layers
act_func_str = "sine"

# for sine activation function in the first layer followed by tanh activation functions
act_func_str = "single-sine"
```
Execute for example:
```
python two_mass_pinn_va.py
```
to run the vanilla version of the PINN.

### Requirements
The code requires tensorflow, scipy, numpy, matplotlib and pickle. We recommend training the PINNs on a GPU.



