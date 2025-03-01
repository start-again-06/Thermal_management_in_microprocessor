ZenteiQ.ai SciML Challenge 🚀

Solve Convection-Dominated Problems using Scientific Machine Learning 🔬

The Problem - Thermal Management in Microprocessors 💻🔥

Problem Statement 📌

The aim of this challenge is to leverage neural network-based partial differential equation (PDE) solvers to predict and prevent thermal damage in next-generation microprocessors before it occurs. In a high-performance computing environment, heat management is crucial for both performance and hardware longevity.

Geometry and Physical Conditions 📏

Consider a 1cm × 1cm processor die, modeled as a unit square domain [0,1] × [0,1].

Heat is generated from specific core regions, with forced cooling provided by two fans:

Primary cooling: y-direction, magnitude 3 cm/s (by = 3)

Secondary cooling: x-direction, magnitude 2 cm/s (bx = 2)

The thermal mixing coefficient is extremely low (ε = 10⁻⁴).

🔥 The Technical Challenge

Participants will develop a Physics-Informed Neural Network (PINN) to solve a Convection-Diffusion (CD) equation, which represents heat transfer phenomena:

where:

u(x,y): Temperature distribution

ε: Thermal diffusion coefficient

(bx, by): Forced cooling field in x-y directions

f(x,y): Heat generation function

🔬 Problem Specifics

Thermal diffusion coefficient: ε = 10⁻⁴ (Extremely low mixing between layers)

Cooling field: (bx, by) = (2,3) (Complex forced cooling patterns)

Heat generation function:


🌡️ Boundary Conditions

The problem is defined on a unit square domain [0,1] × [0,1] with Dirichlet boundary conditions:


This means the temperature is maintained at a reference level (normalized to zero) along all boundaries, representing ideal heat sink conditions.

🚀 How to Run the Model

Install Dependencies 📦

pip install -r requirements.txt

Train the PINN Model 🏋️‍♂️

python train.py --epochs 10000 --lr 0.001 --hidden_layers 4 --neurons_per_layer 64

Evaluate Model 📊

python evaluate.py --model_path saved_models/pinn_model.pth

Visualize Results 🎨

python plot_results.py --data results/output.csv

🤝 How to Contribute

We welcome contributions! To contribute:

Fork the repository 🍴

Create a new branch (feature-branch) 🌱

Commit your changes 💾

Open a pull request 🚀

Feel free to create an issue if you have suggestions! 💡
