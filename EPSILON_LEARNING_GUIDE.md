# Learning Epsilon Directly: Theory and Implementation Guide

## Overview
Currently, **epsilon (ε)** is a **hyperparameter** that you manually choose before training. The professor asks: can we **infer or learn epsilon directly** from data instead?

This guide explains:
1. **Theoretical background** of what ε represents
2. **Where ε appears in the loss function**
3. **How to make ε learnable**
4. **Exact code locations to modify**

---

## Part 1: Theoretical Foundation

### What is Epsilon (ε)?

From the finite-time reach-avoid guarantee:
$$P(\text{RA}(x_0, x_1) \land \text{TimeReach}(x_0, x_1) \leq t) \geq p' - \frac{1}{\varepsilon t}$$

**ε is the minimum expected decrease rate per step:**
- If V(x) decreases by at least ε per step (in expectation), the tail bound on reaching time becomes tighter.
- Larger ε → tighter time bound (reaches faster), but harder to satisfy in training.
- Smaller ε → easier to satisfy, but weaker time guarantee.

### The Loss Encodes ε

In the expected decrease loss, ε appears implicitly:

**Currently (manual ε):**
- The loss tries to enforce: $E[V(x_{k+1})] \leq V(x) - \varepsilon$ (approximately)
- If this holds everywhere, you get a reaching time bound of $P(\text{TimeReach} > t) \leq \frac{1}{\varepsilon t}$

**The loss term is:**
```
Vdiffs = E[V(x_new)] - V(x) + mesh_loss*(K + lip_certificate) + EPS_decrease
loss_exp_decrease = sum(max(0, Vdiffs)) / count
```

**Interpretation:**
- If `Vdiffs ≤ 0`, the decrease condition is satisfied (V decreases by enough).
- The amount it decreases by is related to how negative `Vdiffs` is.
- Currently, you tune `mesh_loss` and `EPS_decrease` to balance this, but **ε is not explicit**.

### How to Infer ε?

**Option 1: Compute ε from empirical measurements**
After training, measure how much V actually decreases per step on average from simulations:
```
ε_empirical = mean(V(x_t) - E[V(x_{t+1})]) over all simulated steps
```
Then use this ε to set p' = p + 1/(ε * t).

**Option 2: Learn ε during training**
Make ε a **learnable scalar** and optimize it alongside V and the policy:
- Loss depends on ε (through the decrease constraint)
- Gradient ∂Loss/∂ε tells you whether to increase or decrease ε
- At convergence, ε reflects the actual minimum achievable decrease rate

**Option 3: Reformulate loss to directly target p and t**
Instead of enforcing V(x_{k+1}) ≤ V(x) - ε implicitly, enforce:
```
p' - 1/(ε * t) ≥ p
```
directly in the loss by:
- Choosing a desired p and t upfront
- Computing ε_required = 1 / (t * (p' - p))
- Penalizing when E[V(x_{k+1})] > V(x) - ε_required

---

## Part 2: Current Code Structure (Where ε appears)

### File: `core/learner.py`

#### Line 102-141: `loss_exp_decrease()` function
```python
def loss_exp_decrease(self, V_state, V_params, x, u, noise_key, probability_bound):
    # Computes E[V(x_new)] given current V, state x, action u, and random noise
    # This is compared against V(x) - epsilon in the loss
```

**Key insight:** This function **computes the expected next-state value**. The loss is built by comparing this against V(x).

#### Line 230-240: Expected decrease loss computation
```python
Vdiffs = jnp.maximum(V_expected - jnp.minimum(V_decrease, jnp.log(3) - jnp.log(1 - probability_bound)) + 
                     mesh_loss * (K + lip_certificate) + EPS_decrease, 0)
```

**Components:**
- `V_expected`: E[V(x_{k+1})] (from loss_exp_decrease)
- `V_decrease`: V(x) clamped to avoid extreme values
- `mesh_loss * (K + lip_certificate)`: correction for discretization and Lipschitz uncertainty
- `EPS_decrease` (≈ 0.01): safety margin to avoid numerical issues

**The gap that needs to become -ε:**
$$\text{Vdiffs} = E[V(x_{k+1})] - V(x) + \text{mesh\_loss} \cdot (K + \text{lip\_certificate}) + \text{EPS\_decrease}$$

For the expected decrease to hold, we need `Vdiffs ≤ -ε`.

#### Line 241-250: Loss aggregation
```python
loss_exp_decrease_mean = expDecr_multiplier * (
    (jnp.sum(Vdiffs_trim, axis=0) + jnp.sum(Vdiffs_cx_trim, axis=0)) \
    / (jnp.sum(...) + 1e-4)
)
```

**This averages all violations.** If ε were explicit, you'd see:
```
loss_exp_decrease_mean = expDecr_multiplier * sum(max(0, E[V(x_new)] - V(x) + mesh_loss*(K+L) + ε))
```

### File: `core/verifier.py`

#### Line 591, 594: Expected decrease verification
```python
C = Vx_center_violations < - np.log(1 - self.args.probability_bound)  # log-RASM
C = Vx_center_violations < 1 / (1 - self.args.probability_bound)      # regular RASM
```

**These check if the decrease condition is satisfied**, implicitly assuming ε was learned/set.

---

## Part 3: Implementation Options

### Option A: Compute ε After Training (Simplest)

**No code changes needed.** After training:

1. Run validation with 100 simulations
2. For each step in each simulation, record V(x_t) and V(x_{t+1})
3. Compute:
```python
decreases = []
for t in range(horizon):
    decrease = V[t] - V[t+1]
    decreases.append(decrease)

epsilon_empirical = np.min(np.mean(decreases, axis=0))  # minimum decrease across steps
```
4. Then set `p' = p + 1/(epsilon_empirical * t)` and re-run verifier with this adjusted p'.

**Pros:** Simple, no code changes  
**Cons:** ε is a byproduct, not optimized for

---

### Option B: Learn ε as a Scalar Parameter (Recommended)

**Modify `core/learner.py`:**

#### Step 1: Add ε to the learner state

In `train_step()` signature (around line 133):
```python
def train_step(self,
               key: jax.Array,
               V_state: TrainState,
               Policy_state: TrainState,
               epsilon_state: TrainState,  # <-- NEW: epsilon as a learnable param
               counterexamples,
               mesh_loss,
               probability_bound,
               expDecr_multiplier):
```

#### Step 2: Define ε as a single scalar network or parameter

Option 2a: **ε as a scalar parameter (simplest)**
```python
# In __init__:
self.epsilon_init_value = 0.01  # starting value

# In train_step, inside loss_fun:
epsilon = jax.lax.stop_gradient(epsilon_state.params['epsilon'])  # retrieve current ε
```

Option 2b: **ε as a small neural network output**
```python
# epsilon_network outputs a single scalar in (0, 1)
epsilon = jnp.clip(epsilon_state.apply_fn(epsilon_params), 1e-4, 0.5)
```

#### Step 3: Modify the expected decrease loss to use ε

Around line 230-240, replace:
```python
# OLD: implicit ε via EPS_decrease margin
Vdiffs = jnp.maximum(V_expected - jnp.minimum(V_decrease, ...) + mesh_loss * (K + lip_certificate) + EPS_decrease, 0)

# NEW: explicit ε
epsilon = 0.01  # or learned parameter
Vdiffs = jnp.maximum(V_expected - jnp.minimum(V_decrease, ...) + mesh_loss * (K + lip_certificate) + epsilon, 0)
```

#### Step 4: Compute gradient w.r.t. ε

```python
# After computing loss, compute gradients
loss, grads = jax.value_and_grad(loss_fun, argnums=(0, 1, 2))(V_params, Policy_params, epsilon_state.params)

# Update epsilon_state using gradient descent
epsilon_state = epsilon_state.apply_gradients(grads=grads[2])
```

#### Step 5: Return updated ε state

```python
return V_grads, Policy_grads, epsilon_grads, infos, key, samples_in_batch, epsilon_state
```

**Key insight:** JAX's autodiff will compute ∂Loss/∂ε. If loss increases when ε increases, ε will decrease (and vice versa), naturally finding the minimum achievable decrease rate.

---

### Option C: Target p and t Directly in Loss

**Most elegant but requires significant refactoring.**

**Idea:**
Instead of:
- Train V to satisfy V(x_{k+1}) ≤ V(x) - ε (with manual ε)

Do:
- Train V to satisfy: p' - 1/(ε*t) ≥ p (with desired p, t as inputs)

**Modifications:**

1. Add p and t as loss inputs:
```python
def train_step(self, ..., desired_p, time_horizon_t):
```

2. Compute required ε:
```python
# What epsilon do we need to achieve p?
epsilon_required = 1.0 / (time_horizon_t * (p_prime - desired_p + 1e-6))
```

3. Enforce this ε in the loss:
```python
Vdiffs = jnp.maximum(V_expected - V_decrease + mesh_loss*(K+L) + epsilon_required, 0)
```

This way, **the loss directly encodes your end goal (p, t)** rather than an intermediate value (ε).

---

## Part 4: Exact Code Locations to Modify

| File | Function | Lines | What to Change | Why |
|------|----------|-------|----------------|-----|
| `core/learner.py` | `__init__` | 20-45 | Add epsilon initialization | Make ε a learnable parameter |
| `core/learner.py` | `train_step` | 133 | Add epsilon_state argument | Pass ε to training step |
| `core/learner.py` | `loss_fun` (nested) | 200-210 | Retrieve/initialize epsilon | Access current ε value |
| `core/learner.py` | `loss_fun` | 230-245 | Use epsilon explicitly in Vdiffs | Replace implicit EPS_decrease |
| `core/learner.py` | `loss_fun` | 260-280 | Same for counterexamples | Consistent across all loss terms |
| `core/learner.py` | `train_step` | ~360 | Compute epsilon gradients | Optimize ε |
| `core/learner.py` | `train_step` return | ~380 | Return updated epsilon_state | Persist ε updates |
| `run.py` | training loop | (search for `train_step` calls) | Pass epsilon_state | Maintain ε across iterations |
| `core/verifier.py` | `__init__` | ~100 | Store learned epsilon | Use ε during verification |
| `validate_certificate.py` | `loss_exp_decrease` | 200-202 | Accept epsilon parameter | Empirical validation with ε |

---

## Part 5: Summary Table

| Approach | Effort | Flexibility | Pros | Cons |
|----------|--------|-------------|------|------|
| **Option A: Compute after training** | Minimal (0 code changes) | Low | Quick to test, diagnostic | ε not optimized, manual adjustment |
| **Option B: Learn ε as scalar** | Medium (modify learner.py) | Medium | ε optimized automatically, clean | Need to initialize, tune learning rate |
| **Option C: Direct p/t encoding** | High (refactor loss) | High | Loss directly targets goal, most principled | Complex, requires careful math |

---

## Part 6: Mathematical Intuition (Why This Works)

When ε is **learned**:
- **If ε is too large:** The loss says "you need to decrease V by a lot," which is hard to satisfy → loss is high → gradient pushes ε down.
- **If ε is too small:** The loss is easily satisfied → gradient tries to push ε up (to be more ambitious).
- **At equilibrium:** ε equals the minimum decrease the network *can* achieve across states. This is optimal because:
  - You get the tightest possible time bound (smallest 1/(ε·t))
  - The training doesn't waste effort on unrealistic ε values

---

## Next Steps

1. **Start with Option A:** Compute epsilon_empirical after your next validation run. Compare it to your manual choices (0.05, 0.01, 0.005). Does it align?

2. **Prototype Option B:** Modify `core/learner.py` to add a learnable scalar ε. Start simple: epsilon as a single `jnp.array([0.01])` parameter with its own optimizer.

3. **Measure impact:** Train with learned ε vs. manual ε (0.01, 0.05). Which achieves better p' − 1/(ε·t)?

4. **Iterate:** If learned ε converges to an unexpected value, audit the loss computation (are there other implicit decreases being masked?).

---

## Questions to Ask Yourself

- **Where is V decreasing the most?** (Should have high ε there)
- **Where is V plateauing?** (Should have low ε there)
- **Is the mesh_loss term hiding an effective ε?** (Lipschitz + mesh corrections might already encode decrease)
- **What happens if you set EPS_decrease = 0?** (Then you'd see pure decrease enforcement)

---

## References
- Finite-time reach-avoid: Paper Section 3, Equation (3)
- Loss formulation: Paper Appendix, Section A.3
- Expected decrease term: Paper Appendix, Equation (A.9)

