# ZenteiQ.ai SciML Challenge  
## Solving Convection-Dominated Problems with Scientific Machine Learning

---

## Challenge Overview: Thermal Management in Microprocessors
In high-performance computing systems, effective thermal management is critical to prevent overheating and ensure reliable operation. This challenge focuses on developing a **Physics-Informed Neural Network (PINN)** to predict temperature distribution across a processor die under forced cooling and localized heat generation.

---

## Problem Statement
A 1 cm × 1 cm processor die is subject to forced convection and internal heat sources. The objective is to solve the **Convection–Diffusion (CD) equation** and accurately predict the temperature field across the domain.

---

## Geometry and Physical Conditions
- **Domain:** $$\([0,1] \times [0,1]\)$$ unit square  
- **Forced Cooling Field:**
  - $$\(b_y = 3\)$$ cm/s (primary fan, y-direction)
  - $$\(b_x = 2\)$$ cm/s (auxiliary fan, x-direction)
- **Thermal Diffusion Coefficient:**  
  $$\(\varepsilon = 10^{-4}\)$$ (convection-dominated regime)

---

## Governing Equation
The temperature distribution $$\(u(x,y)\)$$ satisfies the convection–diffusion equation:

$$
\[
-\varepsilon \nabla^2 u + \mathbf{b} \cdot \nabla u = f(x,y)
\]
$$

where:
- $$\(u(x,y)\)$$ is the temperature field  
- $$\(\varepsilon\)$$ is the diffusion coefficient  
- $$\(\mathbf{b} = (b_x, b_y)\)$$ is the convection velocity  
- $$\(f(x,y)\)$$ represents localized heat generation  

---

## Boundary Conditions
**Dirichlet Boundary Condition:**

$$
\[
u(x,y) = 0 \quad \text{on all boundaries}
\]
$$

---

## Implementation Details

### Model Architecture
- Fully connected neural network  
- Activation function: `tanh`  
- Architecture:  
  - Input layer: 2 neurons $$\((x,y)\)$$
  - Hidden layer: 20 neurons  
  - Output layer: 1 neuron $$\((u)\)$$

### Training Configuration
- Optimizer: Adam  
- Learning rate: $$\(1 \times 10^{-4}\)$$  
- Epochs: 10,000  

### Training Data
- Interior collocation points: 8,000  
- Boundary points: 800  

### Loss Function
- PDE residual loss (physics consistency)  
- Boundary condition loss  

---

## Dataset and Files
- **Heat source distribution:** `test.csv`  
- **Training framework:** TensorFlow with custom loss functions  
- **Submission output:** `y_predict.csv`  

---

## Training Procedure
1. Sample interior and boundary points in the domain.  
2. Compute PDE residuals using automatic differentiation.  
3. Enforce boundary conditions through penalty loss.  
4. Train the PINN for 10,000 epochs.  
5. Generate temperature predictions for submission.

---

## Evaluation Criteria
Submissions are evaluated based on:
- **Prediction Accuracy:** Error against reference temperature values
- **Computational Efficiency:** Training time and resource usage
- **Physical Consistency:** Adherence to the governing PDE

---

## Contributing
Fork the repository and submit a pull request for improvements, optimizations, or alternative modeling approaches.

---

## License
MIT License. Free to use, modify, and distribute for academic and research purposes.

