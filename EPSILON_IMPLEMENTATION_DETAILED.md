# Epsilon Learning: Exact Code Changes (Strategy B)

This document shows **before and after code** for learning ε during training (Strategy B).

---

## Change 1: Add ε to Learner Initialization

**File:** `core/learner.py`  
**Location:** In `__init__` method, after line 45

### BEFORE:
```python
def __init__(self, env, args):
    '''Initialize the learner.'''
    
    self.env = env
    self.linfty = False
    
    # Copy some arguments
    self.auxiliary_loss = args.auxiliary_loss
    self.lambda_lipschitz = args.loss_lipschitz_lambda
    # ... more initialization ...
    
    self.glob_min = 0.1
    self.N_expectation = 16
    
    # Define vectorized functions for loss computation
    self.loss_exp_decrease_vmap = jax.vmap(...)
    return
```

### AFTER:
```python
def __init__(self, env, args):
    '''Initialize the learner.'''
    
    self.env = env
    self.linfty = False
    
    # Copy some arguments
    self.auxiliary_loss = args.auxiliary_loss
    self.lambda_lipschitz = args.loss_lipschitz_lambda
    # ... more initialization ...
    
    self.glob_min = 0.1
    self.N_expectation = 16
    
    # NEW: Initialize epsilon parameter
    self.epsilon_init_value = getattr(args, 'epsilon_init', 0.01)  # Read from args, default 0.01
    self.epsilon_learning_rate = getattr(args, 'epsilon_lr', 1e-3)  # Learning rate for epsilon
    
    # Define vectorized functions for loss computation
    self.loss_exp_decrease_vmap = jax.vmap(...)
    return
```

---

## Change 2: Add epsilon_state to train_step Signature

**File:** `core/learner.py`  
**Location:** `train_step` method, line 133

### BEFORE:
```python
@partial(jax.jit, static_argnums=(0,))
def train_step(self,
               key: jax.Array,
               V_state: TrainState,
               Policy_state: TrainState,
               counterexamples,
               mesh_loss,
               probability_bound,
               expDecr_multiplier):
    '''
    Perform one step of training the neural network.
    '''
```

### AFTER:
```python
@partial(jax.jit, static_argnums=(0,))
def train_step(self,
               key: jax.Array,
               V_state: TrainState,
               Policy_state: TrainState,
               epsilon_state: jax.Array,  # NEW: epsilon parameter(s)
               counterexamples,
               mesh_loss,
               probability_bound,
               expDecr_multiplier):
    '''
    Perform one step of training the neural network.
    
    :param epsilon_state: Current epsilon value(s) (scalar or array)
    '''
```

---

## Change 3: Modify Loss Function to Use ε Explicitly

**File:** `core/learner.py`  
**Location:** Inside `train_step`, in `loss_fun` nested function, around line 195

### BEFORE:
```python
def loss_fun(certificate_params, policy_params):
    
    # Small epsilon used in the initial/unsafe loss terms
    EPS_init = 0.1
    EPS_unsafe = 0.1
    EPS_decrease = self.EPS_decrease  # Fixed value from args
    
    # ... compute losses ...
    
    # Compute E[V(x+)] - V(x), approximated over finite number of noise samples
    if self.exp_certificate:
        Vdiffs = jnp.maximum(
            V_expected - jnp.minimum(V_decrease, jnp.log(3) - jnp.log(1 - probability_bound)) 
            + mesh_loss * (K + lip_certificate) 
            + EPS_decrease,  # ← This is the old implicit epsilon
            0
        )
    else:
        Vdiffs = jnp.maximum(
            V_expected - jnp.minimum(V_decrease, 3 / (1 - probability_bound)) 
            + mesh_loss * (K + lip_certificate) 
            + EPS_decrease,  # ← This is the old implicit epsilon
            0
        )
```

### AFTER:
```python
def loss_fun(certificate_params, policy_params):
    
    # Small epsilon used in the initial/unsafe loss terms
    EPS_init = 0.1
    EPS_unsafe = 0.1
    
    # NEW: Extract learnable epsilon (ensure it stays positive and reasonable)
    epsilon = jnp.clip(epsilon_state, 1e-4, 0.5)  # Keep epsilon in [0.0001, 0.5]
    
    # ... compute losses ...
    
    # Compute E[V(x+)] - V(x), approximated over finite number of noise samples
    if self.exp_certificate:
        Vdiffs = jnp.maximum(
            V_expected - jnp.minimum(V_decrease, jnp.log(3) - jnp.log(1 - probability_bound)) 
            + mesh_loss * (K + lip_certificate) 
            + epsilon,  # ← Now using learnable epsilon
            0
        )
    else:
        Vdiffs = jnp.maximum(
            V_expected - jnp.minimum(V_decrease, 3 / (1 - probability_bound)) 
            + mesh_loss * (K + lip_certificate) 
            + epsilon,  # ← Now using learnable epsilon
            0
        )
```

---

## Change 4: Also Update Loss for Counterexamples

**File:** `core/learner.py`  
**Location:** Around line 275 (in the counterexample section)

### BEFORE:
```python
if len(counterexamples) > 0:
    # ... compute V_cx ...
    
    # Add expected decrease loss
    expDecr_keys_cx = jax.random.split(noise_key, (self.batch_size_counterx, self.N_expectation))
    actions_cx = Policy_state.apply_fn(policy_params, cx_samples)
    V_expected = self.loss_exp_decrease_vmap(V_state, certificate_params, cx_samples, actions_cx,
                                             expDecr_keys_cx, probability_bound)
    if self.exp_certificate:
        Vdiffs_cx = jnp.maximum(
            V_expected - jnp.minimum(V_cx, jnp.log(3) - jnp.log(1 - probability_bound)) 
            + mesh_loss * (K + lip_certificate) 
            + EPS_decrease,  # ← Old
            0
        )
    else:
        Vdiffs_cx = jnp.maximum(
            V_expected - jnp.minimum(V_cx, 3 / (1 - probability_bound)) 
            + mesh_loss * (K + lip_certificate) 
            + EPS_decrease,  # ← Old
            0
        )
```

### AFTER:
```python
if len(counterexamples) > 0:
    # ... compute V_cx ...
    
    # Add expected decrease loss
    expDecr_keys_cx = jax.random.split(noise_key, (self.batch_size_counterx, self.N_expectation))
    actions_cx = Policy_state.apply_fn(policy_params, cx_samples)
    V_expected = self.loss_exp_decrease_vmap(V_state, certificate_params, cx_samples, actions_cx,
                                             expDecr_keys_cx, probability_bound)
    if self.exp_certificate:
        Vdiffs_cx = jnp.maximum(
            V_expected - jnp.minimum(V_cx, jnp.log(3) - jnp.log(1 - probability_bound)) 
            + mesh_loss * (K + lip_certificate) 
            + epsilon,  # ← Now using learnable epsilon
            0
        )
    else:
        Vdiffs_cx = jnp.maximum(
            V_expected - jnp.minimum(V_cx, 3 / (1 - probability_bound)) 
            + mesh_loss * (K + lip_certificate) 
            + epsilon,  # ← Now using learnable epsilon
            0
        )
```

---

## Change 5: Compute Gradients w.r.t. Epsilon

**File:** `core/learner.py`  
**Location:** End of `train_step`, around line 360 where gradients are computed

### BEFORE:
```python
def loss_fun(certificate_params, policy_params):
    # ... loss computation ...
    return loss_total

# Compute gradients w.r.t. V and Policy only
loss_val, grads = jax.value_and_grad(loss_fun, argnums=(0, 1))(
    V_state.params, Policy_state.params
)

loss_val = jnp.asarray(loss_val)
V_grads = grads[0]
Policy_grads = grads[1]
```

### AFTER:
```python
def loss_fun(certificate_params, policy_params):
    # ... loss computation ...
    return loss_total

# NEW: Compute gradients w.r.t. V, Policy, AND epsilon
loss_val, grads = jax.value_and_grad(loss_fun, argnums=(0, 1))(
    V_state.params, Policy_state.params
)

loss_val = jnp.asarray(loss_val)
V_grads = grads[0]
Policy_grads = grads[1]

# NEW: Compute epsilon gradients
_, epsilon_grad = jax.value_and_grad(
    lambda eps: loss_fun(V_state.params, Policy_state.params),
    argnums=None  # Will be replaced by manual computation below
)(epsilon_state)

# NEW: Alternative (simpler): Manual gradient for epsilon
# This requires tracing through loss_fun to see how loss depends on epsilon
# For now, we can compute it by finite differences or use autodiff on just the decrease loss
epsilon_grad = jax.grad(
    lambda eps: jnp.sum(
        jnp.maximum(
            V_expected - jnp.minimum(V_decrease, ...) + mesh_loss * (K + lip_certificate) + eps,
            0
        )
    )
)(epsilon_state)
```

---

## Change 6: Update Epsilon State

**File:** `core/learner.py`  
**Location:** Return statement of `train_step`, around line 380

### BEFORE:
```python
return V_grads, Policy_grads, infos, key, samples_in_batch
```

### AFTER:
```python
# NEW: Update epsilon state using gradient descent
epsilon_state_new = epsilon_state - self.epsilon_learning_rate * epsilon_grad

return V_grads, Policy_grads, epsilon_state_new, infos, key, samples_in_batch
```

---

## Change 7: Initialize Epsilon State in Training Loop

**File:** `core/ppo_jax.py` (or wherever `train_step` is called)  
**Location:** Training loop initialization, around line 100-150

### BEFORE:
```python
# Initialize networks
V_state = create_train_state(...)
Policy_state = create_train_state(...)

# Training loop
for iteration in range(num_iterations):
    key, subkey = jax.random.split(key)
    
    V_state, Policy_state, infos, key, samples = learner.train_step(
        subkey, V_state, Policy_state,
        counterexamples, mesh_loss, args.probability_bound, expDecr_multiplier
    )
```

### AFTER:
```python
# Initialize networks
V_state = create_train_state(...)
Policy_state = create_train_state(...)

# NEW: Initialize epsilon state
epsilon_state = jnp.array([learner.epsilon_init_value])  # Start with initial epsilon

# Training loop
for iteration in range(num_iterations):
    key, subkey = jax.random.split(key)
    
    V_state, Policy_state, epsilon_state, infos, key, samples = learner.train_step(
        subkey, V_state, Policy_state,
        epsilon_state,  # NEW: Pass epsilon state
        counterexamples, mesh_loss, args.probability_bound, expDecr_multiplier
    )
    
    # NEW: Log epsilon every N iterations
    if iteration % 50 == 0 and not args.silent:
        print(f"Iteration {iteration}: epsilon = {float(epsilon_state):.6f}, loss = {float(infos['loss_total']):.4f}")
```

---

## Change 8: Return Epsilon After Training

**File:** `core/ppo_jax.py`  
**Location:** After training loop completes, before returning

### BEFORE:
```python
# Training complete
print(f"Training finished. Final loss: {loss_final:.4f}")
return V_state, Policy_state
```

### AFTER:
```python
# Training complete
print(f"Training finished. Final loss: {loss_final:.4f}")
print(f"Learned epsilon: {float(epsilon_state):.6f}")

# NEW: Save epsilon to checkpoint or config
return V_state, Policy_state, epsilon_state  # or save epsilon to metadata
```

---

## Change 9: Use Learned Epsilon in Verification

**File:** `core/verifier.py` and `validate_certificate.py`  
**Location:** When loading checkpoint, around line 50-100

### BEFORE:
```python
# Load checkpoint
checkpoint_path = Path(args.checkpoint) / 'final_ckpt'
V_state = orbax_checkpointer.restore(checkpoint_path, item=target)['V_state']
```

### AFTER:
```python
# Load checkpoint
checkpoint_path = Path(args.checkpoint) / 'final_ckpt'
V_state = orbax_checkpointer.restore(checkpoint_path, item=target)['V_state']

# NEW: Load learned epsilon (if saved)
try:
    epsilon_learned = orbax_checkpointer.restore(checkpoint_path, item=target).get('epsilon_state', 0.01)
    print(f"- Loaded learned epsilon: {epsilon_learned:.6f}")
except:
    epsilon_learned = 0.01  # Fallback to default
    print(f"- No learned epsilon found, using default: {epsilon_learned:.6f}")
```

---

## Summary of Changes

| Change # | File | Lines Affected | What | Why |
|----------|------|----------------|------|-----|
| 1 | `core/learner.py` | 47-50 | Add epsilon_init_value and epsilon_learning_rate | Initialize learnable ε |
| 2 | `core/learner.py` | 133-137 | Add epsilon_state parameter to train_step | Pass ε into training |
| 3 | `core/learner.py` | 200-210 | Extract epsilon, replace EPS_decrease | Make ε explicit in loss |
| 4 | `core/learner.py` | 275-290 | Same replacement for counterexamples | Consistent ε across loss |
| 5 | `core/learner.py` | 360-375 | Compute epsilon gradients | Optimize ε with gradient descent |
| 6 | `core/learner.py` | 380-385 | Update and return epsilon_state | Persist ε updates |
| 7 | `core/ppo_jax.py` | 120-135 | Initialize epsilon_state, pass to train_step | Integrate ε into training loop |
| 8 | `core/ppo_jax.py` | 200+ | Return epsilon_state after training | Save learned ε |
| 9 | `core/verifier.py`, `validate_certificate.py` | 50-100 | Load epsilon from checkpoint | Use learned ε in verification |

---

## Testing Approach

### Step 1: Add Logging (Minimal Changes)
After Change 6, add a print statement in the training loop to see ε evolve:

```python
if iteration % 10 == 0:
    eps_val = float(epsilon_state[0]) if hasattr(epsilon_state, '__getitem__') else float(epsilon_state)
    print(f"[Iter {iteration:4d}] epsilon={eps_val:.6f}")
```

### Step 2: Validate ε Convergence
- **Expected:** ε stabilizes (doesn't oscillate wildly)
- **Example good output:**
```
[Iter    0] epsilon=0.010000
[Iter   10] epsilon=0.009500
[Iter   20] epsilon=0.009100
[Iter   50] epsilon=0.008800
[Iter  100] epsilon=0.008700  ← converged
[Iter  200] epsilon=0.008700
```

### Step 3: Compare with Manual ε
- Train separately with manual ε = 0.01 (no learning)
- Compare final certificate quality (p', violations, etc.)
- Learned ε should perform better or equivalently

---

## Common Mistakes to Avoid

1. **Don't clip epsilon before using in loss:**
   ```python
   # BAD: Clipping breaks gradients
   epsilon = jnp.clip(epsilon_state, 1e-4, 0.5)
   # Then use epsilon in loss
   
   # GOOD: Clip after gradient computation
   epsilon = jnp.clip(epsilon_state, 1e-4, 0.5)
   # Use epsilon in loss
   ```

2. **Don't forget to update epsilon_state at iteration end:**
   ```python
   # BAD: compute gradients but don't update
   epsilon_grad = ...
   # (missing epsilon_state update)
   
   # GOOD:
   epsilon_grad = ...
   epsilon_state = epsilon_state - learning_rate * epsilon_grad
   ```

3. **Don't mix old EPS_decrease with new epsilon:**
   ```python
   # BAD: Using both
   Vdiffs = ... + EPS_decrease + epsilon
   
   # GOOD: Use only epsilon
   Vdiffs = ... + epsilon
   ```

