# Ellipsoid Method and the Amazing Oracles 🎯

## A 60-Minute Technical Presentation

---

# 📑 Agenda

1. **Introduction** - What is the Ellipsoid Method? 🧭
2. **Core Components** - The Search Space (Ellipsoid) 📦
3. **Cutting Plane Algorithms** - The Engine 🔧
4. **The Amazing Oracles** - Problem-Specific Oracles 🔮
5. **Deep Dive: Ellipsoid Updates** - The Math Behind the Magic 🧮
6. **Real-World Applications** - From Finance to Signal Processing 🌐
7. **Summary & Q&A** ❓

---

# 1. Introduction: The Ellipsoid Method 🧭

## What is the Ellipsoid Method?

The **Ellipsoid Method** is a *polynomial-time* algorithm for convex optimization, introduced by **L.G. Khachiyan in 1979**.

> 🎯 It uses ellipsoids to iteratively reduce the feasible region until an optimal solution is found.

```mermaid
flowchart LR
    A[Initial Ellipsoid] --> B[Query Oracle]
    B --> C{Cut Available?}
    C -->|Yes| D[Update Ellipsoid]
    C -->|No| E[Solution Found!]
    D --> B
    E --> F[Optimal Point]

    style A fill:#e1f5fe
    style E fill:#c8e6c9
    style F fill:#fff9c4
```

## Key Properties

| Property | Description |
|----------|-------------|
| ⏱️ **Time Complexity** | $O(n^4 L)$ where $L$ is input bits |
| 📊 **Iterations** | Polynomial in problem dimension |
| 🎯 **Optimality** | Guarantees $\epsilon$-optimal solution |
| 🔄 **Robustness** | Works with only subgradient access |

---

# 2. Core Components: The Search Space 📦

## The `Ell` Class

```python
from ellalgo.ell import Ell
import numpy as np

# Create ellipsoid with radius 1.0 centered at origin
ell = Ell(1.0, np.array([0.0, 0.0]))
```

### Visual Representation

```mermaid
graph TD
    subgraph Ellipsoid Structure
        A[_xc: Center] --> B["xc\(\\) method"]
        C[_mq: Shape Matrix] --> B
        D[_kappa: Scaling] --> B
        E[_tsq: Distance] --> B["tsq\(\\) method"]
    end

    B --> F[Get Current Point]
    E --> G[Check Convergence]

    style A fill:#ffecb3
    style C fill:#ffecb3
    style D fill:#ffecb3
    style E fill:#ffecb3
```

### Ellipsoid Parameters

- **$x_c$**: Center point (current best solution candidate) 📍
- **$M$**: Shape matrix (defines ellipsoid geometry) 🧭
- **$\kappa$**: Scaling factor 🔢
- **$\tau^2$**: Distance measure to optimal point 📏

---

# 3. Cutting Plane Algorithms 🔧

## The Main Algorithm

```python
from ellalgo.cutting_plane import cutting_plane_feas, cutting_plane_optim
from ellalgo.ell_config import Options

# Feasibility problem
x, niter = cutting_plane_feas(oracle, space, Options())

# Optimization problem
x, gamma, niter = cutting_plane_optim(oracle, space, gamma_init, Options())
```

## Algorithm Flow

```mermaid
sequenceDiagram
    participant C as Cutting Plane
    participant S as Search Space
    participant O as Oracle

    Note over C: Start with initial ellipsoid
    loop for each iteration
        C->>S: Get center point xc
        S-->>C: Return xc

        C->>O: assess_feas(xc) / assess_optim(xc, gamma)
        O-->>C: Return cut (or None)

        alt cut is not None
            C->>S: update_bias_cut(cut) / update_central_cut(cut)
            S-->>C: Return status
        else
            Note over C: Solution found!
        end
    end
```

---

# 4. Types of Cuts 🗡️

## Deep Cut (Bias Cut)

```python
# Deep cut - cuts through ellipsoid
ell.update_bias_cut((gradient, beta))
```

The deep cut equation:

$$
g^T(x - x_c) + \beta \leq 0
$$

Where $\beta$ controls cut position (not through center).

## Central Cut

```python
# Central cut - passes through center
ell.update_central_cut((gradient, 0.0))
```

```mermaid
graph LR
    subgraph Cut Types
        A[Ellipsoid] --> B[Deep Cut]
        A --> C[Central Cut]
        A --> D[Parallel Cut]
    end

    B --> B1[β ≠ 0, asymmetric]
    C --> C1[β = 0, through center]
    D --> D1[Two parallel hyperplanes]

    style B1 fill:#e8eaf6
    style C1 fill:#e8eaf6
    style D1 fill:#e8eaf6
```

## Parallel Cut

```python
# Parallel cut - two constraints
# Format: (gradient, (beta_lower, beta_upper))
ell.update_bias_cut((gradient, (beta0, beta1)))
```

Useful when you have constraints like:
$$
a^T x \leq b_1 \quad \text{and} \quad -a^T x \leq -b_2
$$

---

# 5. The Amazing Oracles 🔮

## Oracle Interfaces

```mermaid
classDiagram
    class OracleFeas {
        <<abstract>>
        +assess_feas(x_center) Optional[Cut]
    }

    class OracleOptim {
        <<abstract>>
        +assess_optim(x_center, gamma) Tuple[Cut, Optional[float]]
    }

    class OracleOptimQ {
        <<abstract>>
        +assess_optim_q(x_center, gamma, retry) Tuple[Cut, Array, Optional[float], bool]
    }

    OracleFeas <|-- LMIOracle
    OracleFeas <|-- ProfitOracle
    OracleOptim <|-- LowpassOracle
    OracleOptim <|-- ProfitOracle
    OracleOptimQ <|-- ProfitQOracle
```

## 5.1 LMI Oracle - Linear Matrix Inequalities 📐

```python
from ellalgo.oracles.lmi_oracle import LMIOracle

# LMI: B - (F1*x1 + F2*x2 + ...) ⪰ 0
oracle = LMIOracle([F1, F2, ..., Fn], B)

cut = oracle.assess_feas(xc)
# Returns (gradient, violation) if infeasible
```

### Mathematical Background

The LMI constraint:

$$
B - \sum_{i=1}^n F_i x_i \succeq 0
$$

Using **LDLT factorization** to check positive semidefiniteness:

```mermaid
graph TD
    A[Input xc] --> B["Construct Matrix M(xc)"]
    B --> C[LDLT Factorization]
    C --> D{Success?}
    D -->|Yes| E["Feasible: M ⪰ 0"]
    D -->|No| F[Compute Cut]
    F --> G[Subgradient g]
    F --> H[witness vector]

    style E fill:#c8e6c9
    style F fill:#ffcdd2
```

### LDLT Manager

```python
from ellalgo.oracles.ldlt_mgr import LDLTMgr

ldlt_mgr = LDLTMgr(matrix_size)
if ldlt_mgr.factor(get_elem):
    # Matrix is PSD
else:
    witness = ldlt_mgr.witness()
    g = ldlt_mgr.sym_quad(Fk)  # subgradient
```

---

## 5.2 Profit Oracle - Economic Optimization 💰

```python
from ellalgo.oracles.profit_oracle import ProfitOracle

# Cobb-Douglas production function
# max p*A*y1^α*y2^β - v1*y1 - v2*y2
# s.t. y1 ≤ k

oracle = ProfitOracle(
    params=(unit_price, scale, limit),  # (p, A, k)
    elasticities=np.array([α, β]),      # [α, β]
    price_out=np.array([v1, v2])        # [v1, v2]
)
```

### Mathematical Model

**Production Function:**
$$
f(y) = p \cdot A \cdot y_1^\alpha \cdot y_2^\beta
$$

**Profit:**
$$
\pi(y) = p A y_1^\alpha y_2^\beta - v_1 y_1 - v_2 y_2
$$

**Constraint:**
$$
y_1 \leq k
$$

### Solution in Log-Space

```mermaid
graph LR
    A[y in ℝ] --> B["log-space: x = log(y)"]
    B --> C[Oracle Assessment]
    C --> D{Feasible?}
    D -->|No| E[Feasibility Cut]
    D -->|Yes| F[Compute Profit]
    F --> G[Optimality Cut]

    style E fill:#ffcdd2
    style G fill:#c8e6c9
```

### Variations

```python
# Robust Profit Oracle - handles uncertainty
from ellalgo.oracles.profit_oracle import ProfitRbOracle

oracle_rb = ProfitRbOracle(
    params=(unit_price, scale, limit),
    elasticities=np.array([α, β]),
    price_out=np.array([v1, v2]),
    vparams=(ε1, ε2, ε3, ε4, ε5)  # uncertainty params
)

# Discrete Profit Oracle - integer inputs
from ellalgo.oracles.profit_oracle import ProfitQOracle

oracle_q = ProfitQOracle(params, elasticities, price_out)
# Maximizes profit with y ∈ ℕ²
```

---

## 5.3 Lowpass Oracle - FIR Filter Design 📡

```python
from ellalgo.oracles.lowpass_oracle import LowpassOracle, create_lowpass_case

# Design FIR lowpass filter
oracle = LowpassOracle(
    ndim=48,      # Number of coefficients
    wpass=0.12,   # Passband edge (normalized)
    wstop=0.20,   # Stopband edge
    lp_sq=0.99,   # Lower passband bound (squared)
    up_sq=1.01,   # Upper passband bound (squared)
    sp_sq=0.01    # Stopband bound (squared)
)

# Or use default case
oracle = create_lowpass_case()
```

### Filter Design Problem

**Objective:** Minimize maximum stopband response

$$
\min_x \max_{w \in \text{stopband}} |H(w)|
$$

**Subject to:**
$$
\frac{1}{\delta} \leq |H(w)| \leq \delta \quad \text{for } w \in \text{passband}
$$
$$
|H(w)| \leq \text{stopband bound} \quad \text{for } w \in \text{stopband}
$$

### Spectral Factorization

```mermaid
graph TD
    A[Impulse Response h] --> B[Spectral Factorization]
    B --> C[Autocorrelation r]
    C --> D["Frequency Response |H(w)|²"]

    style A fill:#e1f5fe
    style D fill:#c8e6c9
```

Using **Kolmogorov 1939** method:

```python
from ellalgo.oracles.spectral_fact import spectral_fact

r = np.array([1.0, 0.5, 0.2])  # Autocorrelation
h = spectral_fact(r)  # Minimum-phase impulse response
```

### Constraint Checking

```python
# Passband constraints
if response > upper_bound:
    return gradient, (violation_lower, violation_upper)

# Stopband constraints
if response > stopband_limit:
    return gradient, (stopband_violation, response)
```

---

# 6. Deep Dive: Ellipsoid Updates 🧮

## The Update Formulas

Given a cut $(g, \beta)$:

### Step 1: Compute Key Values

$$
\begin{aligned}
\omega &= g^T M g \\
\rho &= \frac{\kappa \beta + \sqrt{\kappa^2 + \omega (\kappa + \beta)^2}}{\omega + \kappa} \\
\sigma &= \frac{\kappa + \beta}{\kappa + \beta + \rho \omega} \\
\delta &= \frac{\kappa + \beta + \rho \omega}{\kappa (n + 1)}
\end{aligned}
$$

### Step 2: Update Parameters

```python
# In Python (from ell.py)
grad_t = M @ grad              # M * g
omega = grad.dot(grad_t)       # g^T * M * g
rho, sigma, delta = result     # From EllCalc

# Update center
xc_new = xc - (rho / omega) * grad_t

# Update shape matrix
M_new = M - (sigma / omega) * outer(grad_t, grad_t)

# Update scaling
kappa_new = kappa * delta
```

### Visual Interpretation

```mermaid
graph TD
    subgraph Ellipsoid Update
        A[Original Ellipsoid] --> B[Apply Cut]
        B --> C[Shrink & Reshape]
        C --> D[New Ellipsoid]
    end

    B --> E[Cut Hyperplane]
    E --> F[Exclude Infeasible Region]
    F --> G[Volume Reduction]

    style A fill:#e1f5fe
    style D fill:#c8e6c9
```

### Volume Reduction Guarantee

The ellipsoid method guarantees at least **$e^{-1/(n+1)}$** volume reduction per iteration!

For $n=10$: $\approx 0.95^{11} \approx 57\%$ reduction

---

# 7. Real-World Applications 🌐

## Application Overview

| Application | Oracle | Problem Type |
|-------------|--------|--------------|
| 📐 Control Systems | LMIOracle | Stability verification |
| 💰 Portfolio Optimization | ProfitOracle | Resource allocation |
| 📡 Signal Processing | LowpassOracle | Filter design |
| 🔬 Robust Optimization | ProfitRbOracle | Uncertainty handling |

## LMI in Control Systems

```python
# Verify: A^T P + P A + Q < 0 has solution P ⪰ 0
# Lyapunov stability condition

F1 = ...  # Coefficients
F2 = ...
B = ...   # Constraint matrix

oracle = LMIOracle([F1, F2], B)
cut = oracle.assess_feas(P)
```

## Filter Design Example

```mermaid
graph LR
    subgraph Design Process
        A[Initialize r] --> B[Ellipsoid Method]
        B --> C{Converged?}
        C -->|No| D[Spectral Factorization]
        D --> B
        C -->|Yes| E[Optimal Filter]
    end

    style E fill:#c8e6c9
```

---

# 8. Code Example: Putting It All Together 🛠️

## Complete Example: Filter Design

```python
import numpy as np
from ellalgo.ell import Ell
from ellalgo.cutting_plane import cutting_plane_optim
from ellalgo.ell_config import Options
from ellalgo.oracles.lowpass_oracle import create_lowpass_case

# Create oracle and initial ellipsoid
oracle = create_lowpass_case(48)
x0 = np.zeros(48)
space = Ell(10.0, x0)

# Run optimization
options = Options()
options.max_iters = 2000
options.tolerance = 1e-8

x, gamma, niter = cutting_plane_optim(oracle, space, 0.5, options)

print(f"Iterations: {niter}")
print(f"Stopband attenuation: {gamma:.4f}")
```

---

# 9. Summary 📝

## Key Takeaways

1. 🎯 **Ellipsoid Method**: Polynomial-time algorithm for convex optimization
2. 📦 **Search Space**: Ellipsoid defined by center, shape matrix, and scaling
3. 🗡️ **Cuts**: Deep cuts, central cuts, parallel cuts for different strategies
4. 🔮 **Oracles**: Problem-specific implementations that provide cutting planes
5. 📐 **LMI Oracle**: Linear Matrix Inequalities for control & systems
6. 💰 **Profit Oracle**: Economic optimization with Cobb-Douglas production
7. 📡 **Lowpass Oracle**: FIR filter design via spectral factorization

## Architecture Diagram

```mermaid
graph TD
    subgraph ellalgo
        A[cutting_plane.py] --> B[ell.py]
        B --> C[ell_calc.py]
        C --> D[ell_calc_core.py]

        A --> E[oracles/]
        E --> E1[LMI Oracle]
        E --> E2[Profit Oracle]
        E --> E3[Lowpass Oracle]
    end

    style A fill:#fff3e0
    style B fill:#fff3e0
    style C fill:#fff3e0
    style E1 fill:#e8eaf6
    style E2 fill:#e8eaf6
    style E3 fill:#e8eaf6
```

---

# ❓ Questions?

## Resources

- 📚 Documentation: [ellalgo.readthedocs.io](https://ellalgo.readthedocs.io/)
- 💻 Source Code: [github.com/luk036/ellalgo](https://github.com/luk036/ellalgo)
- 📖 Paper: Khachiyan, L.G. (1979) "Polynomial algorithms in linear programming"

---

# Thank You! 🎉

*Made with ❤️ using Python + NumPy*
