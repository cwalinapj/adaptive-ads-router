# LAM Training Input Pipeline

To move from a "black box" LLM-based system to a self-contained Large Action Model (LAM), we must convert raw container logs into a high-fidelity training corpus.

The LAM Training Input Pipeline doesn't just store what happened; it stores the **delta between states** and the **intent behind the move**.

## 1. Dataset Assembler (The "Lineage" Collector)

The assembler joins four distinct data sources into a single Training Tuple. This is triggered every time a container is "tombstoned" (regenerated).

### The Tuple Structure

```
T = (S_base, V_mutation, C_user, R_reward)
```

| Component | Description |
|-----------|-------------|
| **S_base** | The "Before" state (DOM snapshot + CSS variables) |
| **V_mutation** | The direction-vector (what the LLM tried to do) |
| **C_user** | The behavioral embeddings (how users moved) |
| **R_reward** | The final scalar (Thompson-sampled mean score) |

## 2. Feature Normalization & Semantic Hashing

Raw HTML is too noisy for an offline training loop. We must normalize features into a **Structural Hash**:

### DOM Flattening
Reduce the HTML to a depth-limited tree of semantic components:
```
HERO > CTA > SOCIAL_PROOF
```

### Color Space Mapping
Convert HEX/RGB into a relative contrast ratio:
```
CTA_CONTRAST: 0.85
```

### Token Distillation
Use a small encoder (BERT-base) to turn the landing page copy into a fixed **768-dimensional semantic centroid**.

## 3. Direction-Vector Alignment

This is the most critical component. We align the LLM's **Intent** with the **Observed Structural Delta**.

If the LLM claimed it was increasing "Trust," but the Structural Diff shows it actually just changed a button color, the training record is penalized for **"Intent-Action Mismatch."**

### Mutation Vector Logic

```
V_direction = [Urgency, Scarcity, Trust, Friction, Aesthetic_Complexity]
```

Each value is a float from **-1.0 to +1.0**.

| Example | Meaning |
|---------|---------|
| Trust +0.8 | Adding testimonials/logos |
| Friction -0.5 | Shortening a form or removing a scroll-gate |

## 4. Offline Training Loop (The LAM Forge)

The training loop operates on a **Policy-Gradient** style update. We want the LAM to predict the V_direction that maximizes R_reward given S_base and C_user.

```python
# Conceptual Training Step
for batch in dataset.assemble_batches(size=32):
    # Predict the expected reward for a mutation
    predicted_reward = lam_model.predict(batch.base_state, batch.mutation_vector)
    
    # Calculate Loss: (Predicted Reward - Actual Thompson Mean)
    loss = mse_loss(predicted_reward, batch.actual_reward)
    
    # Backpropagate to sharpen the LAM's "intuition" on what moves work
    loss.backward()
```

## 5. Transition: From First-100 to First-1000

| Metric | First-100 (LLM Dominant) | First-1000 (LAM Emergent) |
|--------|--------------------------|---------------------------|
| **Generation** | External LLM (Zero-shot) | LAM (Few-shot + Distilled) |
| **Routing** | Pure Thompson Sampling | Contextual Bandit (Neural-Weighted) |
| **Verification** | Diff Enforcement (Hard) | Diff Enforcement (Soft/Neural) |
| **Data Usage** | Log collection | Continuous fine-tuning |

## 6. Cross-Site Abstraction Transfer

Once the LAM has seen 10 sites, it begins to notice patterns:

> "Social Proof in the Hero" works for **Financial Services** but fails for **Direct-to-Consumer Luxury**.

The LAM learns:
- **Industry-specific priors**
- **Device-specific optimizations**
- **Audience segment patterns**

This is the emergence of **transfer learning** in conversion optimization.
