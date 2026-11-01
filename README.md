# Quantum Machine Learning: Classical vs. Variational Quantum Classifiers (VQC & DR-QNN)
### Hackathon Project for Tinkerspace & QuantumX Community Hackathon
**Location:** Cochin, Kerala | **Date:** November 1, 2026

A unified research, benchmark, and interactive demonstration platform comparing **Traditional Machine Learning Classifiers**, a **Baseline Variational Quantum Classifier (VQC)**, and an **Enhanced Data Re-Uploading Quantum Neural Network (Qiskit-DRQNN)** across clinical healthcare analytics and non-linear geometric topological manifolds.


---

## 1. Executive Summary & Core Results

Developed for the **Tinkerspace & QuantumX Community Hackathon** in Cochin, Kerala, this project tackles the practical boundaries of Quantum Machine Learning (QML). We investigate:
* *Can shallow quantum circuits classify complex non-linear structures without exploding parameter counts?*
* *How can the Barren Plateau problem (exponentially vanishing gradients) be mitigated on hardware-efficient ansaetze?*
* *How does multi-layer Data Re-Uploading transform a parameterized circuit into a Universal Function Approximator?*

```mermaid
flowchart TD
    A["Input Dataset (Clinical Pima Indians / Non-Linear Manifolds)"] --> B["Preprocessing & Stratified Partitioning"]
    B --> C["Traditional ML Pipeline (5 Estimators)"]
    B --> D["Baseline Quantum VQC (4 Qubits, Single Map, COBYLA)"]
    B --> E["Enhanced Quantum DR-QNN (Data Re-Uploading, Local Observables, Adam)"]
    C --> F["Model Benchmarking Suite (Accuracy, F1, ROC-AUC, Latency)"]
    D --> F
    E --> F
    F --> G["Streamlit Interactive Research Laboratory"]
```

### Key Empirical Findings

| Model Architecture | Concentric Circles Acc | Two Moons Acc | Clinical Diabetes ROC-AUC | Training Speed | Gradient Signal Health |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Traditional Extra Trees / Random Forest** | 98.5% | 97.2% | **0.831** | < 0.5s | Classical Backprop |
| **Traditional Logistic Regression (Linear)** | 47.9% *(Fails)* | 89.6% | 0.812 | < 0.05s | Convex Landscape |
| **Baseline Quantum VQC (COBYLA, Global Z)** | 79.2% | 85.4% | 0.575 *(Weak)* | 158.0s | Vanishing ($\mathcal{O}(2^{-n})$) |
| **Enhanced Qiskit-DRQNN (Ours)** | **93.8%** | **100.0%** | **0.804** | **1.42s** | **Healthy ($\Omega(1/\text{poly}(n))$)** |

---

## 2. Complete Technology Stack

* **Quantum Computing & Simulation Framework:**
  * `Qiskit 2.6.1`: Native quantum circuit construction, `QuantumCircuit`, `ParameterVector`, `SparsePauliOp`.
  * `Qiskit-Aer 0.18.2`: Multi-threaded C++ quantum statevector backend and noisy circuit simulation.
  * `Qiskit Machine Learning 0.9.1`: Primitive-based quantum neural networks (`EstimatorQNN`, `SamplerQNN`).
* **High-Speed Simulation Acceleration:**
  * Custom `FastVectorizedQiskitEngine` in pure `NumPy 2.x` / C: tensor contractions (`np.einsum`) evolving batches of quantum states $(B, 2, 2, \dots, 2)$ simultaneously in milliseconds (**>1,000 samples/sec**, $\sim 70\times$ faster).
* **Classical Machine Learning & Statistics:**
  * `Scikit-Learn 1.6+`: Logistic Regression, Random Forest, Gradient Boosting, Extra Trees, Gaussian Naive Bayes, Stratified K-Fold CV, Mutual Information feature selection.
  * `SciPy 1.13+`: Mathematical optimization and statistical metrics.
* **Interactive Dashboard & Visualization:**
  * `Streamlit 1.40+`: Modern interactive analytical dashboard.
  * `Plotly Express 5.0+`: Interactive 2D decision boundary contours, ROC curves, confusion matrices, and training loss plots.
  * `Matplotlib 3.8+`: Publication-grade diagnostic figures.
* **Data Processing & Engineering:**
  * `Pandas 2.2+`: DataFrame manipulation and reproducible split serializations.
  * `Joblib`: Pipeline and model persistence.

---

## 3. Environment Setup (`venv` & `requirements.txt`): Why & For What We Use It

### Why Use a Dedicated Virtual Environment (`qiskit-env`)?
1. **Dependency Isolation:** Prevents polluting the global operating system Python environment and eliminates library conflicts between incompatible package versions.
2. **Binary Extension Compatibility:** Quantum packages such as `qiskit-aer` and `rustworkx` rely on pre-compiled native C++ binaries. An isolated virtual environment ensures that native DLLs are linked cleanly without collision.
3. **Reproducibility for Hackathon Judges & Evaluators:** Allows any reviewer or team member to replicate the exact runtime environment on their machine with zero "works on my machine" issues.

### Why Use `requirements.txt` and What is Pinned?
* Qiskit underwent a major architectural rewrite from Qiskit 0.x/1.x to Qiskit 2.x (removing legacy modules like `qiskit.opflow` and switching to V2 Primitives).
* Our `requirements.txt` locks the exact tested dependencies:
  * `qiskit>=2.0,<3.0` and `qiskit-aer>=0.16,<1.0`: Modern Qiskit 2.x quantum core and simulator.
  * `qiskit-machine-learning>=0.9.0`: Native Qiskit 2.x machine learning integrations.
  * `numpy>=2.0,<3.0`, `scipy>=1.13,<2.0`, `scikit-learn>=1.6,<2.0`: Modern numerical and modeling foundations.
  * `streamlit>=1.40,<2.0` and `plotly>=5.0,<8.0`: High-performance interactive UI and plotting.

---

## 4. Outer Validation Protocol & Proof of Generalization (`outer_validation/`)

In-sample performance is often misleading in quantum machine learning due to circuit overfitting or flat loss landscapes. We established an **`outer_validation/`** folder containing concrete visual proof and metrics validating that our model generalizes to unseen test points and held-out medical records.

### Proof 1: Two Moons Non-Convex Boundary (`outer_validation/drqnn_moons_validation.png`)
![Two Moons Outer Validation](outer_validation/drqnn_moons_validation.png)
* **Generalization Score:** **100.0% accuracy** on unseen test coordinates (gold stars).
* **Smooth Manifold:** The quantum circuit warps the decision boundary cleanly around both interlocking crescents.
* **Gradient Signal:** Gradient norm remains bounded at $\sim 0.20$, verifying healthy parameter updates.

### Proof 2: Concentric Circles Radial Enclosure (`outer_validation/drqnn_circles_validation.png`)
![Concentric Circles Outer Validation](outer_validation/drqnn_circles_validation.png)
* **Generalization Score:** **93.8% accuracy** on unseen test data.
* **Classical Linear Failure:** Classical linear models achieved only **47.9%** (failed).
* **Multi-Harmonic Fourier Fit:** Data Re-Uploading enables the circuit to construct closed circular separation contours.

### Proof 3: Clinical Diabetes Outer Validation (`outer_validation/drqnn_diabetes_validation.png`)
![Clinical Diabetes Outer Validation](outer_validation/drqnn_diabetes_validation.png)
* **Held-Out Test ROC-AUC:** **0.804**, proving generalization to real-world physiological indicators with zero data leakage.

*For full protocol details, see [`outer_validation/OUTER_VALIDATION_REPORT.md`](outer_validation/OUTER_VALIDATION_REPORT.md).*

---

## 5. Methodology & Steps Proposed

### Step 1: Clinical Dataset Ingestion & Validation
* Dataset: **Pima Indians Diabetes Dataset** (500 validated records, 8 clinical features).
* Zero-value validation: Validated physiological indicators (Glucose, Blood Pressure, Insulin, BMI) before processing.
* Stratified 80/20 train/test split (random state 42) ensuring identical class ratios in training (400 records) and testing (100 records).

### Step 2: Traditional Estimator Benchmarking
* Five diverse model families trained via scikit-learn pipelines with standard scaling and median imputation.
* Hyperparameter optimization via 5-fold Stratified Cross-Validation strictly on training data.
* Decision threshold optimization derived from out-of-fold training probabilities to balance recall and precision without test data leakage.

### Step 3: Baseline Quantum Classifier (VQC) Diagnosis
* A 4-qubit hardware-light circuit: $R_y$ feature encoding, $R_{zz}$ pairwise interactions, and nearest-neighbour $CZ$ entangling ladder.
* Optimized via COBYLA minimizing binary cross-entropy on a subset of 80 records.
* **Identified Bottlenecks:**
  1. *Barren Plateaus:* Measuring global parity $Z^{\otimes 4}$ leads to exponentially vanishing gradient variances ($\mathcal{O}(2^{-n})$).
  2. *Single-Map Expressivity Limit:* Encoding data once limits the model to first-order Fourier frequencies ($e^{\pm i x}$).
  3. *Latency Overhead:* Sequential evaluation required over 2.5 minutes for just 50 COBYLA steps.

### Step 4: Enhanced Quantum Architecture (Qiskit-DRQNN)
To solve these bottlenecks, the Enhanced DR-QNN implements four core architectural solutions:
1. **Interleaved Data Re-Uploading:** Re-injects classical inputs $R_y(x_l)$ between variational blocks across $L$ layers. Mathematically synthesizes an $L$-harmonic Fourier series:
   $$f(x) = \sum_{k \in \{-L, \dots, L\}^d} c_k e^{i k \cdot x}$$
   elevating the quantum circuit into a **Universal Function Approximator**.
2. **Local Pauli-Z Observables:** Readout is performed via local single-qubit observables ($\langle Z_0 \rangle, \langle Z_1 \rangle$) using Qiskit's `SparsePauliOp`. Local observables guarantee a polynomial gradient variance lower bound:
   $$\mathrm{Var}\left[\frac{\partial \mathcal{L}}{\partial \theta}\right] \in \Omega\left(\frac{1}{\mathrm{poly}(n)}\right)$$
   maintaining **$4.4\times$ stronger gradient flow**.
3. **Vectorized Batch Tensor Simulation:** Evaluates parameter-shift gradients $\frac{\partial \langle Z \rangle}{\partial \theta} = \frac{1}{2}(\langle Z(\theta + \pi/2) \rangle - \langle Z(\theta - \pi/2) \rangle)$ across the entire mini-batch in parallel, cutting training time from **158s to 1.4s**.
4. **Calibrated Softmax Temperature Readout:** Maps raw expectations $[-1, 1]$ to calibrated class probabilities using learnable temperature $\gamma$ and bias $\beta$:
   $$\text{logit}_k = \gamma_k \langle Z_k \rangle + \beta_k, \quad P(Y=k|X) = \frac{e^{\text{logit}_k}}{\sum_j e^{\text{logit}_j}}$$
   ensuring well-calibrated confidence and smooth cross-entropy optimization.

### Step 5: Geometric Non-Linear Manifold Validation
* Benchmarked on three non-linearly separable geometric datasets:
  * **Concentric Circles:** Radial non-linearity where linear models score ~47.9% (fails); DR-QNN achieves **93.8%**.
  * **Interlocking Two Moons:** Non-convex manifold boundary; DR-QNN achieves **100.0%**.
  * **4-Quadrant XOR:** Continuous parity problem; DR-QNN achieves **96.5%**.

### Step 6: Unified Streamlit Laboratory Integration
* Integrated all pipelines into a cohesive multi-page research dashboard (`Dashboard/app.py`) featuring live training controls, real-time patient risk sliders, 2D decision boundary heatmaps, and Qiskit circuit inspection.

---

## 6. Qiskit Quantum Circuit Architecture

The native parameterized circuit synthesized by `build_drqnn_circuit()`:

```
     ┌─────────────┐┌──────────────┐┌──────────────┐      ┌─────────────┐┌──────────────┐┌──────────────┐      
q_0: ┤ Ry(x_l0[0]) ├┤ Rz(theta[0]) ├┤ Ry(theta[1]) ├─■──■─┤ Ry(x_l1[0]) ├┤ Rz(theta[4]) ├┤ Ry(theta[5]) ├─■──■─
     ├─────────────┤├──────────────┤├──────────────┤ │  │ ├─────────────┤├──────────────┤├──────────────┤ │  │ 
q_1: ┤ Ry(x_l0[1]) ├┤ Rz(theta[2]) ├┤ Ry(theta[3]) ├─■──■─┤ Ry(x_l1[1]) ├┤ Rz(theta[6]) ├┤ Ry(theta[7]) ├─■──■─
     └─────────────┘└──────────────┘└──────────────┘      └─────────────┘└──────────────┘└──────────────┘      
```

* **Gate Functions:**
  * `Ry(x_l[q])`: Classical feature injection rotating qubit state around the Y-axis.
  * `Rz(theta)` & `Ry(theta)`: Variational rotation parameters placing qubits into parameterized superpositions.
  * `CZ(q, target)`: Entangling Controlled-Z ladder generating cross-qubit quantum phase correlations.
  * Local Measurements: `SparsePauliOp` measuring single-qubit observables `['IZ', 'ZI']`.

---

## 7. Detailed Conclusions & Real-World Applicability

### Scientific Conclusions
1. **Classical vs. Quantum on Tabular Data:** Classical ensemble models (Random Forest, Extra Trees) remain highly effective, fast, and robust for standard tabular spreadsheets. Quantum neural networks are not meant to replace classical ML on basic 2D tables.
2. **Expressivity via Data Re-Uploading:** The Fourier frequency spectrum accessible to a quantum circuit is bounded strictly by the number of times data is re-entered. Data Re-Uploading is essential for practical QML.
3. **Barren Plateau Resolution:** Local cost functions are mandatory for scalable quantum training. Global parity observables must be avoided in deep parameterized quantum circuits.
4. **Computational Latency:** Tensor-vectorized batch simulation bridges the gap between quantum theoretical research and real-time experimentation.

### High-Value Commercial & Industrial Applications
* **Quantum Chemistry & Molecular Drug Discovery (Biotech/Pharma):** Predicting molecular docking and receptor affinity where electron orbitals are inherently quantum mechanical.
* **Correlated Financial Tail-Risk & Fraud Detection (Fintech):** Modeling complex entangled risk across thousands of market assets during extreme financial stress events.
* **Materials Science & Battery Design (Clean Energy/EV):** Rapid screening of crystal lattice structures and solid-state electrolytes.
* **Accelerated QML Middleware:** Providing high-speed simulation engines and barren plateau mitigators for enterprise quantum research teams.

---

## 8. Installation & Quickstart (Windows PowerShell)

Ensure you are in the project root directory:

```powershell
# 1. Activate the included virtual environment
.\qiskit-env\Scripts\Activate.ps1

# 2. Update pip and install project dependencies
.\qiskit-env\Scripts\python.exe -m pip install -r requirements.txt
```

### Running the End-to-End Workflow

```powershell
# Step 1: Environment and package smoke test
.\qiskit-env\Scripts\python.exe Test.py

# Step 2: Prepare reproducible stratified splits
.\qiskit-env\Scripts\python.exe -m shared.preprocess

# Step 3: Train and evaluate traditional classical models
.\qiskit-env\Scripts\python.exe -m traditional.train_traditional

# Step 4: Train baseline quantum model
.\qiskit-env\Scripts\python.exe -m quantum.train_quantum

# Step 5: Run full test suite (7 unittests)
.\qiskit-env\Scripts\python.exe -m unittest discover -s tests -v

# Step 6: Launch the Streamlit Interactive Dashboard
.\qiskit-env\Scripts\python.exe -m streamlit run Dashboard\app.py
```

---

## 9. Project Directory Structure

```
tinkerspace/
├── Dashboard/
│   └── app.py                      # Unified Streamlit research dashboard
├── data/
│   ├── pima_diabetes.csv           # Validated source clinical dataset
│   └── processed/                  # Reproducible train/test splits and metadata
├── outer_validation/               # External validation proofs & benchmark artifacts
│   ├── drqnn_circles_validation.png
│   ├── drqnn_moons_validation.png
│   ├── drqnn_diabetes_validation.png
│   └── OUTER_VALIDATION_REPORT.md  # Detailed outer validation protocol & results
├── quantum/
│   ├── drqnn.py                    # Enhanced Data Re-Uploading Quantum Neural Network
│   ├── model.py                    # Baseline 4-qubit quantum circuit and inference helpers
│   ├── train_quantum.py            # Baseline quantum model training script
│   ├── models/                     # Saved baseline quantum model pickle
│   └── results/                    # Quantum evaluation metrics and diagrams
├── shared/
│   ├── metrics.py                  # Standardized evaluation metrics and curve plotting
│   └── preprocess.py               # Dataset validation and stratified train/test splitters
├── tests/
│   ├── test_dashboard.py           # Dashboard navigation and rendering unit tests
│   └── test_pipeline.py            # Preprocessing and model pipeline tests
├── traditional/
│   ├── train_traditional.py        # Classical model training and grid search tuning
│   ├── models/                     # Saved classical model joblib pipelines
│   └── results/                    # Classical metrics, curves, and comparisons
├── .gitignore                      # Excludes venvs, caches, and temporary artifacts
├── qiskit-env/                     # Dedicated virtual environment with Qiskit 2.x & ML
├── requirements.txt                # Pinned dependencies
├── Test.py                         # Environment validation script
└── README.md                       # Comprehensive hackathon project documentation
```

---

Link To Test :
https://quantum-ml-tinker.streamlit.app/


## 10. Academic & Educational Disclaimer

This project is intended exclusively for educational demonstration, academic research, and algorithmic benchmarking for the **Tinkerspace & QuantumX Community Hackathon**. It is **not** a clinical diagnostic tool or medical device.
