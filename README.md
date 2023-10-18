# AAAD
 The source code of auto-adversarial attack and defense

 Authors: Jialiang Sun, Wen Yao, Tingsong Jiang, Chao Li, Xiaoqian Chen

<img src = 'figures/AAAD.png?raw=true'/>

## Abstract
 Due to the urgent need of the robustness of deep neural networks (DNN),
  numerous existing open-sourced tools or platforms are developed to evaluate the robustness of DNN models by ensembling the majority of adversarial attack or defense algorithms. Unfortunately, current platforms can neither optimize the DNN architectures nor the configuration of adversarial attacks to further enhance the model robustness or the performance of adversarial attacks. To alleviate these problems, in this paper, we propose a novel platform called auto-adversarial attack and defense (A3D), which can help search for robust neural network architectures and efficient adversarial attacks.
   A3D integrates multiple neural architecture search methods to find robust architectures under different robustness evaluation metrics. 
   Besides, we provide multiple optimization algorithms to search for efficient adversarial attacks. In addition, we combine auto-adversarial attack and defense together to form a unified framework. Among auto adversarial defense, the searched efficient attack can be used as the new robustness evaluation to further enhance the robustness. In auto-adversarial attack, the searched robust architectures can be utilized as the threat model to help find stronger adversarial attacks. Experiments on CIFAR10, CIFAR100, and ImageNet datasets demonstrate the feasibility and effectiveness of the proposed platform.

arxiv: https://arxiv.org/abs/2102.11860v1

## The introduction of the core modules of A3D:
```
AAA: provide different search algorithms to search for the near-optimal attacks
optimizer_adv: provide different NAS algorithms to search for robust architectures under different evaluations
noise:the visualization of different noises
plot: the visualization of architectures
retrain: retrain the searched architectures
eval_robustness:evaluate the model robustness
```



Our implementations are based on following papers:
