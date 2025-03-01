🔥 ZenteiQ.ai SciML Challenge

🚀 Solve Convection-Dominated Problems with Scientific Machine Learning

🏆 The Challenge: Thermal Management in Microprocessors

In high-performance computing, managing heat in next-generation microprocessors is crucial to prevent thermal damage and maintain optimal performance. Your task is to develop a Physics-Informed Neural Network (PINN) to predict and control temperature distribution across a processor die.

📌 Problem Statement

A 1cm × 1cm processor die is subject to forced cooling and localized heat generation. Your model must solve the Convection-Diffusion (CD) equation to predict temperature variations across this domain.

🔬 Geometry & Physical Conditions

Domain: [0,1] × [0,1] unit square (processor die)

Cooling field:

Primary fan: by = 3 cm/s (y-direction)

Auxiliary fan: bx = 2 cm/s (x-direction)

Thermal diffusion coefficient: ε = 10⁻⁴ (low thermal mixing)

🏗️ Convection-Diffusion Equation



where:

 = Temperature distribution

 = Thermal diffusion coefficient

 = Forced cooling field

 = Heat generation function

🔥 Heat Generation Function



🔲 Boundary Conditions (Dirichlet)

 for all 

🛠️ Implementation Details

📌 Model Setup

Neural Network Architecture: Fully connected with tanh activation

Layers: Input (2 neurons) → Hidden (20 neurons) → Output (1 neuron)

Optimizer: Adam (learning rate = 1e-4)

Loss Function: Combination of PDE residual loss & boundary loss

Training Data:

Interior points: 8000

Boundary points: 800

📂 Code & Dataset

Dataset: Heat source distribution provided in test.csv

Training: Implemented in TensorFlow with custom loss functions

Submission: Generate predicted temperature values in y_predict.csv

🚀 Training Procedure

Generate random points for interior and boundary conditions.

Define PINN loss as:

PDE loss (ensuring physical consistency)

Boundary loss (enforcing boundary conditions)

Train the model for 10,000 epochs.

Evaluate results and generate submission file.

📢 How to Run

🔧 Requirements

Install dependencies using:

pip install tensorflow numpy pandas matplotlib

▶️ Run the Training Script

python train.py

📤 Generate Predictions

python generate_submission.py

🏅 Evaluation Criteria

Your submission will be evaluated based on:

Accuracy: Error between predicted and true temperature values.

Computational Efficiency: Training time and resource utilization.

Physical Consistency: Adherence to the governing PDE.

🤝 Contribute

Fork the repository and submit a pull request if you have improvements!

🌟 Good luck, and may the best SciML engineer win! 🚀🔥
