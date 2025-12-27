"""
Diff Enforcer - Validates container mutations stay within bounds.
"""

class DiffEnforcer:
    def __init__(self, phase: str):
        self.phase = phase
    
    def validate(self, base_html: str, new_html: str) -> bool:
        """Validate that the diff between base and new HTML is acceptable."""
        struct_diff = self._dom_complexity(base_html, new_html)
        hard_limit = 0.50 if self.phase == "explore" else 0.30
        if struct_diff > hard_limit:
            return False
        return self._verify_required_elements(new_html)
    
    def _dom_complexity(self, a: str, b: str) -> float:
        """Calculate DOM complexity difference (normalized 0-1)."""
        a_tags = a.count("<")
        b_tags = b.count("<")
        return abs(a_tags - b_tags) / max(a_tags, b_tags, 1)
    
    def _verify_required_elements(self, html: str) -> bool:
        """Verify required elements are present."""
        required = ["googletagmanager.com", "id=\"main-cta\""]
        return all(r in html for r in required)


class RegimeDiffEnforcer:
    """
    Diff enforcement by regime (First-100, Middle, Neural).
    """
    
    ALLOWED_CHANGES = {
        "first_100": [
            "cta_color", "cta_text", "cta_size", "cta_position",
            "headline", "subheadline", "hero_image",
            "testimonial_order", "form_fields", "layout", "spacing"
        ],
        "middle": [
            "cta_color", "cta_text", "cta_size",
            "headline", "testimonial_order", "spacing"
        ],
        "neural": [
            "cta_text", "headline", "testimonial_order"
        ]
    }
    
    FORBIDDEN_CHANGES = [
        "logo", "brand_colors", "legal_text", "pricing", "privacy_policy"
    ]
    
    MAX_DIFF_SCORE = {
        "first_100": 0.8,
        "middle": 0.4,
        "neural": 0.1
    }
    
    def __init__(self, regime: str = "first_100"):
        self.regime = regime
    
    def validate(self, changes: dict, hypothesis: str = None) -> tuple:
        """
        Validate proposed changes.
        
        Returns:
            (allowed, violations, diff_score)
        """
        violations = []
        
        if not hypothesis:
            violations.append("Hypothesis required for mutation")
        
        allowed = self.ALLOWED_CHANGES.get(self.regime, [])
        max_score = self.MAX_DIFF_SCORE.get(self.regime, 0.5)
        
        for change_type in changes.keys():
            if change_type in self.FORBIDDEN_CHANGES:
                violations.append(f"Forbidden change: {change_type}")
            elif change_type not in allowed:
                violations.append(f"Change not allowed in {self.regime}: {change_type}")
        
        diff_score = self._calculate_diff_score(changes)
        
        if diff_score > max_score:
            violations.append(f"Diff score {diff_score:.2f} exceeds max {max_score}")
        
        return len(violations) == 0, violations, diff_score
    
    def _calculate_diff_score(self, changes: dict) -> float:
        """Calculate diff score based on changes."""
        weights = {
            "cta_color": 0.1, "cta_text": 0.15, "cta_size": 0.1,
            "cta_position": 0.2, "headline": 0.25, "subheadline": 0.15,
            "hero_image": 0.3, "testimonial_order": 0.1, "form_fields": 0.25,
            "layout": 0.5, "spacing": 0.1,
            "logo": 1.0, "brand_colors": 1.0, "legal_text": 1.0,
            "pricing": 1.0, "privacy_policy": 1.0,
        }
        total = sum(weights.get(k, 0.2) for k in changes.keys())
        return min(total, 1.0)
    
    def get_allowed_changes(self) -> list:
        """Get allowed change types for current regime."""
        return self.ALLOWED_CHANGES.get(self.regime, [])
    
    def get_max_diff_score(self) -> float:
        """Get max diff score for current regime."""
        return self.MAX_DIFF_SCORE.get(self.regime, 0.5)
