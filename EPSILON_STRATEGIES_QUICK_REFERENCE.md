# Three Strategies for Learning Epsilon: Quick Reference

## Strategy Overview

### STRATEGY A: Compute ε After Training (Diagnostic)
```
┌─────────────┐
│ Train V, π  │  (with manual ε = 0.01)
└──────┬──────┘
       │
       ▼
┌─────────────────────────────────────┐
│ Run 100 validation simulations      │
│ Track V(x_t) - V(x_{t+1}) per step │
└──────┬──────────────────────────────┘
       │
       ▼
┌──────────────────────────────────┐
│ ε_empirical = min(mean(decreases)) │  ◄── Your ε was too high/low
└──────┬───────────────────────────┘
       │
       ▼
┌──────────────────────────────────────┐
│ Rerun verifier with p' = p + 1/(ε*t)│
└──────────────────────────────────────┘

Code changes: 0
Time to implement: <1 hour
When to use: First step, diagnostic
```

---

### STRATEGY B: Learn ε During Training (Recommended)
```
┌─────────────────────────────────────────────────────┐
│ Learner State = {V_params, Policy_params, epsilon}  │
└──────────┬──────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────┐
│ Loss = loss_init + loss_unsafe +                     │
│        loss_exp_decrease(ε)  ◄── ε affects loss    │
└──────────┬───────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ ∂Loss/∂ε computed by JAX autodiff    │
│ ε_new = ε - learning_rate * ∂Loss/∂ε │
└──────────┬──────────────────────────┘
           │
           ▼  (repeat for N iterations)
           │
┌──────────┴──────────────────────────────┐
│ ε converges to optimal value             │
│ (minimum achievable decrease rate)       │
└─────────────────────────────────────────┘

Code changes: ~30-50 lines in core/learner.py, core/ppo_jax.py
Time to implement: 2-3 hours
When to use: Production, most principled approach
```

---

### STRATEGY C: Target (p, t) Directly in Loss (Most Elegant)
```
Input: desired p = 0.99, time_horizon t = 10*T

┌─────────────────────────────────────┐
│ Compute ε_required = 1/(t*(p'-p))   │
│ (What ε do we need to achieve p?)   │
└──────────┬────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ Enforce in loss:                     │
│ E[V(x')] ≤ V(x) - ε_required         │
│ (So that p' - 1/(ε*t) ≥ p)          │
└──────────┬────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────┐
│ p' naturally emerges from training   │
│ (no manual p choice needed)          │
└──────────────────────────────────────┘

Code changes: ~50-100 lines, refactor loss, add p, t args
Time to implement: 3-4 hours
When to use: Future work, if you want to directly target guarantees
```

---

## Decision Tree: Which Strategy?

```
                          START
                            │
                    ┌───────┴────────┐
                    │                │
            "Is ε   │              No │
           manual   │                │
            tuning  │         Do you have time?
           working?"│         (2+ hours)
            Yes     │
                 ┌──┴─────────┐
                 │            │
                 ▼            ▼
           → USE NOTHING   ┌──────────┐
             (you're done) │  Time?  │
                           │ 3+ hrs  │
                           └──┬──┬──┘
                              │  │
                         Yes  │  │ No
                              ▼  ▼
                           ┌─────────────────┐
                           │ Want elegant?   │
                           │  math-direct    │
                           └─────────────────┘
                              │       │
                         Yes  │       │  No
                              ▼       ▼
                         STRATEGY C STRATEGY B
                    (p, t encoding)  (learn ε)
```

---

## Code Locations for Each Strategy

### Strategy A (Compute ε)

**In `validate_certificate.py` after line 410 (after simulations):**

```python
# Add this after printing avg_steps and avg_time:
if len(steps) > 0:
    # Collect V(x_t) and V(x_{t+1}) for all simulations
    # This requires modifying empirical_reachability to return more details
    # For now, use approximate method:
    epsilon_estimate = avg_steps / (time_horizon * 2)  # rough proxy
    p_required = probability_bound + 1.0 / (epsilon_estimate * time_horizon)
    print(f"- Estimated epsilon: {epsilon_estimate:.4f}")
    print(f"- Required p' for {time_horizon} steps: {p_required:.4f}")
```

---

### Strategy B (Learn ε)

**In `core/learner.py` around line 200 (inside loss_fun):**

```python
# OLD (line 240):
EPS_decrease = self.EPS_decrease

# NEW:
# Retrieve epsilon from learner state (assumed passed in)
# For now, hardcode for testing:
epsilon_learnable = 0.01  # Will be replaced with gradient-updated value

# Then modify Vdiffs:
Vdiffs = jnp.maximum(
    V_expected - jnp.minimum(V_decrease, ...) 
    + mesh_loss * (K + lip_certificate) 
    + epsilon_learnable,  # ← Use this instead of EPS_decrease
    0
)
```

**In `core/ppo_jax.py` or training loop:**

```python
# Create optimizer for epsilon
epsilon_init = jnp.array([0.01])
epsilon_state = optimizer.init(epsilon_init)

# Inside training loop:
V_state, Policy_state, epsilon_state = learner.train_step(
    key, V_state, Policy_state, 
    epsilon_state,  # ← Pass epsilon state
    counterexamples, mesh_loss, probability_bound, expDecr_multiplier
)
```

---

### Strategy C (Direct p, t)

**Requires refactoring of train_step signature and loss computation. Start after Strategy B works.**

---

## Recommended Sequence

### Phase 1 (This week):
1. Run current setup with your chosen (ε, t) combinations
2. Use **Strategy A** to compute epsilon_empirical
3. Compare empirical ε to your manual choices
4. Report findings to professor: "Manual ε = 0.01, empirical ε ≈ X"

### Phase 2 (Next week):
1. Implement **Strategy B** (learn ε as scalar)
2. Train with learned ε
3. Compare final ε to your manual choices
4. Measure p' − 1/(ε·t) and see if it's tighter

### Phase 3 (Optional):
1. If confident, implement **Strategy C** (encode p, t directly)
2. Experimental validation

---

## Testing Checklist for Strategy B

- [ ] ε initializes to 0.01 (or your choice)
- [ ] ε is included in gradient computation
- [ ] ε updates visible in training logs (print ε every 10 iterations)
- [ ] ε converges (doesn't oscillate wildly)
- [ ] Final ε is in reasonable range (0.001–0.1)
- [ ] Loss decreases monotonically (or becomes stable)
- [ ] Verification still passes with final ε
- [ ] p' = p + 1/(final_ε * t) is achievable (< 1.0)

---

## Example Output (What Success Looks Like)

### After Phase 1 (Strategy A):
```
Manual ε tested: [0.05, 0.01, 0.005]
Empirical ε from simulations: 0.018
→ Best manual choice (0.01) was too low
→ Suggests trying ε ≈ 0.015–0.02 next
```

### After Phase 2 (Strategy B):
```
Iteration 0: ε = 0.01000, loss = 2.45
Iteration 100: ε = 0.01523, loss = 1.89
Iteration 500: ε = 0.01687, loss = 1.42
Iteration 1000: ε = 0.01701, loss = 1.39
→ ε converged, loss stable
→ Learned ε = 0.017 aligns with empirical!
```

