"""Container Diff Validator for MCP."""


class ContainerValidator:
    ALLOWED_CHANGES = {
        "first_100": ["cta_color", "cta_text", "cta_size", "cta_position",
                      "headline", "subheadline", "hero_image",
                      "testimonial_order", "form_fields", "layout", "spacing"],
        "middle": ["cta_color", "cta_text", "cta_size", "headline",
                   "testimonial_order", "spacing"],
        "neural": ["cta_text", "headline", "testimonial_order"]
    }
    
    FORBIDDEN_CHANGES = ["logo", "brand_colors", "legal_text", "pricing", "privacy_policy"]
    
    MAX_DIFF_SCORE = {"first_100": 0.8, "middle": 0.4, "neural": 0.1}
    
    def validate(self, regime: str, changes: dict, hypothesis: str = None) -> tuple:
        violations = []
        if not hypothesis:
            violations.append("Hypothesis required")
        
        allowed = self.ALLOWED_CHANGES.get(regime, [])
        max_score = self.MAX_DIFF_SCORE.get(regime, 0.5)
        
        for change_type in changes.keys():
            if change_type in self.FORBIDDEN_CHANGES:
                violations.append(f"Forbidden: {change_type}")
            elif change_type not in allowed:
                violations.append(f"Not allowed in {regime}: {change_type}")
        
        diff_score = self._calculate_diff_score(changes)
        if diff_score > max_score:
            violations.append(f"Diff score {diff_score:.2f} > {max_score}")
        
        return len(violations) == 0, violations, diff_score
    
    def _calculate_diff_score(self, changes: dict) -> float:
        weights = {"cta_color": 0.1, "cta_text": 0.15, "headline": 0.25,
                   "hero_image": 0.3, "layout": 0.5, "logo": 1.0}
        return min(sum(weights.get(k, 0.2) for k in changes.keys()), 1.0)
    
    def get_allowed_changes(self, regime: str) -> list:
        return self.ALLOWED_CHANGES.get(regime, [])
    
    def get_max_diff_score(self, regime: str) -> float:
        return self.MAX_DIFF_SCORE.get(regime, 0.5)
