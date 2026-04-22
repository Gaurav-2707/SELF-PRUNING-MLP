# Sparse Neural Network via Learnable Gates

## 1. Introduction

This project explores **structured pruning** in neural networks using a learnable gating mechanism. Each weight is multiplied by a **sigmoid gate**, allowing the model to learn which connections are important and which can be removed.

To encourage sparsity, an **L1 penalty** is applied to the gate values during training. The goal is to study how different regularization strengths (λ) affect:

- Model accuracy  
- Network sparsity  
- Gate distributions  

---

## 2. Why L1 Penalty on Sigmoid Gates Encourages Sparsity

Each weight $w$ is modulated by a gate:

$\tilde{w} = w \cdot \sigma(g)$

where $\sigma(g) \in (0,1)$.

We add an L1 penalty:

$\mathcal{L}_{sparsity} = \lambda \sum |\sigma(g)|$

### Key intuition:

- The L1 penalty **pushes gate values toward zero**
- Since sigmoid outputs are bounded \([0,1]\), minimizing L1 encourages:
  - Many gates → **close to 0 (pruned weights)**
  - Some gates → **stay high (important weights)**

### Result:
- The network naturally becomes **sparse**
- Unimportant connections are effectively removed:
  $w \cdot \sigma(g) \approx 0$

This creates a **soft pruning mechanism** that is fully differentiable.

---

## 3. Experimental Results

### Summary Table

| Lambda | Test Accuracy (%) | Sparsity Level (%) |
|--------|------------------|--------------------|
| 0.00010 | 57.70 | 97.88 |
| 0.00050 | 57.71 | 99.69 |
| 0.00100 | 57.42 | 99.85 |

---

### Observations

- Increasing λ:
  - ✅ Increases sparsity significantly  
  - ❌ Slightly reduces accuracy  

- Even at extreme sparsity (>99%), the model:
  - Still retains **reasonable performance (~57%)**
  - Shows strong robustness to pruning  

- λ = **0.0005** provides the best balance:
  - Highest sparsity (~99.7%)
  - Maintains peak accuracy  

---

## 4. Training Behavior

The training process shows consistent trends across different λ values:
![training curves](res/training_curves.png)

- **Training Loss**
  - Rapid drop in early epochs  
  - Higher λ starts with larger loss due to stronger regularization  

- **Test Accuracy**
  - Gradual improvement across all λ values  
  - Converges to similar performance levels  

- **Sparsity**
  - Rapid increase in the first few epochs  
  - Plateaus near:
    - ~98% (λ=0.0001)
    - ~99.8% (λ=0.001)

---

## 5. Gate Distribution Analysis (Best Model)
![gate distribution 1](res/gate_distribution_0.0001.png)
![gate distribution 2](res/gate_distribution_0.0005.png)
![gate distribution 3](res/gate_distribution_0.001.png)


We select **λ = 0.0005** as the best model.

### Interpretation:

- Clear **spike at 0** → many weights are pruned  
- Remaining gates cluster **away from 0** → important connections preserved  
- This bimodal behavior confirms:
  - Successful **automatic pruning**
  - Effective separation of useful vs redundant weights  

---

## 6. Discussion: λ Trade-off

| λ Value | Effect |
|--------|--------|
| Low (0.0001) | Less sparsity, slightly better stability |
| Medium (0.0005) | Best trade-off between sparsity and accuracy |
| High (0.001) | Extreme sparsity, slight accuracy drop |

### Key Insight:
- The model can tolerate **extreme sparsity (>99%)**
- However, too large λ:
  - Over-penalizes gates  
  - Removes useful connections  

---

## 7. Conclusion

This project demonstrates that:

- Learnable sigmoid gates + L1 regularization provide an effective pruning mechanism  
- The network can self-prune to **>99% sparsity**  
- Performance degradation is relatively small  
- Proper tuning of λ is critical  
- Training and evaluation pipeline  
- Visualization of gate distributions  
