"""
Thompson Sampling Bandit for Adaptive Routing.
"""

import numpy as np
from dataclasses import dataclass
import json


@dataclass
class Arm:
    """A single bandit arm (page variant)."""
    page_id: str
    alpha: int = 1
    beta: int = 1
    
    @property
    def total(self) -> int:
        return (self.alpha - 1) + (self.beta - 1)
    
    @property
    def conversions(self) -> int:
        return self.alpha - 1
    
    @property
    def mean(self) -> float:
        return self.alpha / (self.alpha + self.beta)
    
    def sample(self) -> float:
        return np.random.beta(self.alpha, self.beta)
    
    def update(self, converted: bool):
        if converted:
            self.alpha += 1
        else:
            self.beta += 1
    
    def to_dict(self) -> dict:
        return {"page_id": self.page_id, "alpha": self.alpha, "beta": self.beta}
    
    @classmethod
    def from_dict(cls, data: dict) -> "Arm":
        return cls(page_id=data["page_id"], alpha=data["alpha"], beta=data["beta"])


class ThompsonSamplingBandit:
    """Thompson Sampling bandit with regime awareness."""
    
    def __init__(
        self,
        site_id: str,
        first_100_threshold: int = 100,
        neural_threshold: int = 1000,
        confidence_threshold: float = 0.95
    ):
        self.site_id = site_id
        self.arms = []
        self.first_100_threshold = first_100_threshold
        self.neural_threshold = neural_threshold
        self.confidence_threshold = confidence_threshold
    
    @property
    def total_sessions(self) -> int:
        return sum(arm.total for arm in self.arms)
    
    @property
    def regime(self) -> str:
        total = self.total_sessions
        if total < self.first_100_threshold:
            return "first_100"
        elif total < self.neural_threshold:
            return "middle"
        return "neural"
    
    def add_arm(self, page_id: str):
        self.arms.append(Arm(page_id=page_id))
    
    def select_arm(self) -> tuple:
        if not self.arms:
            raise ValueError("No arms available")
        if len(self.arms) == 1:
            return 0, self.arms[0].page_id
        samples = [arm.sample() for arm in self.arms]
        best_idx = int(np.argmax(samples))
        return best_idx, self.arms[best_idx].page_id
    
    def update(self, page_id: str, converted: bool):
        for arm in self.arms:
            if arm.page_id == page_id:
                arm.update(converted)
                return
        raise ValueError(f"Unknown page_id: {page_id}")
    
    def get_winner(self):
        if len(self.arms) < 2:
            return None
        for arm in self.arms:
            if arm.total < 3:
                return None
        n_samples = 10000
        samples = np.array([
            np.random.beta(arm.alpha, arm.beta, size=n_samples)
            for arm in self.arms
        ])
        best_probs = []
        for i in range(len(self.arms)):
            is_best = np.all(samples[i] >= samples, axis=0)
            best_probs.append(np.mean(is_best))
        max_prob = max(best_probs)
        best_idx = best_probs.index(max_prob)
        if max_prob >= self.confidence_threshold:
            return self.arms[best_idx].page_id, max_prob
        return None
    
    def get_loser(self):
        winner = self.get_winner()
        if winner is None:
            return None
        winner_id, _ = winner
        for arm in self.arms:
            if arm.page_id != winner_id:
                return arm.page_id
        return None
    
    def get_divergence(self) -> float:
        if len(self.arms) < 2:
            return 0.0
        rates = [arm.mean for arm in self.arms]
        return max(rates) - min(rates)
    
    def get_stats(self) -> dict:
        return {
            "site_id": self.site_id,
            "regime": self.regime,
            "total_sessions": self.total_sessions,
            "arms": [
                {"page_id": arm.page_id, "sessions": arm.total,
                 "conversions": arm.conversions, "rate": f"{arm.mean:.2%}"}
                for arm in self.arms
            ],
            "divergence": self.get_divergence(),
            "winner": self.get_winner()
        }
    
    def to_dict(self) -> dict:
        return {
            "site_id": self.site_id,
            "arms": [arm.to_dict() for arm in self.arms],
            "first_100_threshold": self.first_100_threshold,
            "neural_threshold": self.neural_threshold,
            "confidence_threshold": self.confidence_threshold
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ThompsonSamplingBandit":
        bandit = cls(
            site_id=data["site_id"],
            first_100_threshold=data.get("first_100_threshold", 100),
            neural_threshold=data.get("neural_threshold", 1000),
            confidence_threshold=data.get("confidence_threshold", 0.95)
        )
        bandit.arms = [Arm.from_dict(a) for a in data.get("arms", [])]
        return bandit
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> "ThompsonSamplingBandit":
        return cls.from_dict(json.loads(json_str))
