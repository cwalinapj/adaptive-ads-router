# Diff Metrics (D)

We measure drift across three specific vectors. If any single vector exceeds its threshold, the mutation is flagged.

## A. Structural DOM Diff (Complexity Gap)

We compare the tree depth and element count.

**Formula:**
```
D_struct = |E_base - E_new| / E_base
```

**Thresholds:**
- > 30% change is blocked in "Cosmetic" phase
- > 60% change is blocked in "Explore" phase

## B. Semantic Drift (Vector Distance)

We embed the text of Page A and Page B into a 1536-dimensional space and calculate the Cosine Distance.

**Threshold:** > 0.20

If the LLM changes the page from "Low-cost Insurance" to "Free Gift Cards," the drift is caught here.

## C. Visual Weight Shift (CSS Diff)

We calculate the ratio of "Above-the-Fold" pixel density for specific CSS classes (e.g., `.btn-primary`).

**Threshold:** If the primary CTA:
- Moves more than 400px, OR
- Changes color contrast significantly

→ Flagged for manual review or auto-rejection.

## Summary

| Metric | Formula | Cosmetic Threshold | Explore Threshold |
|--------|---------|-------------------|-------------------|
| Structural DOM | `|E_base - E_new| / E_base` | 30% | 60% |
| Semantic Drift | Cosine distance (1536-dim) | 0.20 | 0.20 |
| Visual Weight | CTA position + contrast | 400px / significant | 400px / significant |

## Implementation

```python
class DiffMetrics:
    def structural_diff(self, base_html: str, new_html: str) -> float:
        """D_struct = |E_base - E_new| / E_base"""
        e_base = base_html.count("<")
        e_new = new_html.count("<")
        return abs(e_base - e_new) / max(e_base, 1)
    
    def semantic_drift(self, base_text: str, new_text: str) -> float:
        """Cosine distance in embedding space"""
        # Embed both texts to 1536-dim vectors
        # Return cosine distance
        pass
    
    def visual_weight_shift(self, base_css: dict, new_css: dict) -> dict:
        """CTA position and contrast changes"""
        # Calculate pixel shift
        # Calculate contrast ratio change
        pass
    
    def validate(self, base: Page, new: Page, phase: str) -> bool:
        struct_threshold = 0.30 if phase == "cosmetic" else 0.60
        
        if self.structural_diff(base.html, new.html) > struct_threshold:
            return False
        if self.semantic_drift(base.text, new.text) > 0.20:
            return False
        if self.visual_weight_shift(base.css, new.css)["cta_shift"] > 400:
            return False
        
        return True
```
