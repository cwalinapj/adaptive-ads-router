# LAM Roadmap

## Large Action Model Integration

### Phase 1: Current (Data Collection)
- Thompson Sampling Bandit
- LLM-Guided Mutations (First-100)
- Session Neural State Collection
- Tombstone Records

### Phase 2: Prediction Model
- Train model to predict conversion probability
- DeepFM / Transformer architecture
- Input: page features, proposed changes, visitor context
- Output: P(conversion | page, change, visitor)

### Phase 3: Generation Model
- Generate mutation hypotheses automatically
- Learn from successful mutations
- Learn from tombstone failures
- Output: ranked list of proposed changes

### Phase 4: Full LAM
- Autonomous optimization
- Minimal human oversight
- Generation → Prediction → Validation → Execution loop

## Timeline

| Phase | Target | Key Deliverables |
|-------|--------|------------------|
| Phase 1 | Now | Session collection, tombstones |
| Phase 2 | Q2 | Prediction model |
| Phase 3 | Q3 | Generation model |
| Phase 4 | Q4 | Full LAM autonomy |

## Success Metrics

- Prediction AUC > 0.75
- 20% reduction in failed mutations
- 15%+ conversion lift vs baseline
