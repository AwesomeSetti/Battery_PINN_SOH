# Battery PINN SOH Modeling

This repository contains a physics-informed neural network (PINN) framework for **battery state-of-health (SOH) estimation**, with extensions that incorporate **temperature effects** and **operating-condition awareness**.  
The project is developed as part of a graduate-level research effort in battery modeling and machine learning.

---

## Project Motivation

Accurate estimation of battery SOH is critical for battery management systems, lifecycle prediction, and safety-aware operation.  
Purely data-driven models often achieve good accuracy but lack physical consistency, while purely physics-based models can be difficult to calibrate and generalize.

This project bridges the two by embedding **degradation dynamics as physics-informed constraints** within a neural network framework.

---

## Core Contributions

- Physics-Informed Neural Network (PINN) formulation for SOH evolution  
- Dual-network architecture separating **state estimation** and **latent degradation dynamics**
- Cell-level generalization across long cycling histories
- Extension to **temperature-aware degradation modeling**
- Conditioning on charging and operational descriptors via StressNetPlus**
- Modular design enabling future extensions (multi-objective health indicators, MOE, etc.)

---

## Repository Structure


> **Note:** Raw datasets, large result files, and trained model weights are intentionally excluded.

---

## Methodology Overview

### PINN Formulation
- The SOH trajectory is modeled as a continuous function of cycling time and extracted features.
- A neural dynamics network enforces physically consistent degradation behavior via a PDE-inspired residual loss.
- Data loss and physics loss are jointly optimized during training.

### Temperature-Aware Extension
- Temperature is incorporated as an explicit model input.
- An Arrhenius-style scaling is used to modulate degradation rates.
- Conditioning variables allow protocol-dependent stress effects without requiring closed-form equations.

### StressNet / StressNetPlus
- Operational descriptors (e.g., charging characteristics) are embedded through a conditioning network.
- This enables flexible adaptation across different cycling regimes while preserving the core degradation structure.

---

## Datasets

Due to licensing and size constraints, datasets are **not included** in this repository.

The project references:
- Michigan fast-formation dataset
- CALCE accelerated cycle life dataset


Scripts assume datasets are prepared locally in user-defined paths.

---

## Intended Use

This repository is intended for:
- Research replication and extension
- Educational exploration of PINNs for battery modeling
- Methodological reference for physics-informed ML in energy systems

It is **not** intended as a plug-and-play production BMS solution.

---

## Future Work

Planned and suggested extensions include:
- Multi-objective health indicators (capacity + resistance + power fade)
- Mixture-of-Experts (MoE) architectures for heterogeneous degradation modes
- Probabilistic uncertainty quantification
- Coupling with physics-based simulators (e.g., PyBaMM)

---

## Author

**Soroush Setayeshpour**  
PhD Student, Mechanical Engineering  
The University of Texas at Dallas  

---

## Disclaimer

This is a research codebase under active development.  
Results, implementations, and assumptions may evolve as part of ongoing academic work.
