"""
Thompson Sampling Bandit for Adaptive Routing.

Includes:
- Basic Thompson Sampling (ThompsonSamplingBandit)
- Contextual Thompson Sampling (ContextualBandit) - uses device/geo/time features
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json


@dataclass
class Context:
    """Visitor context for contextual bandit."""
    device: str = "desktop"  # mobile, desktop, tablet
    geo: str = "us"          # country code
    hour: int = 12           # hour of day (0-23)

    def to_bucket(self) -> str:
        """Convert context to bucket key for stratification."""
        # Bucket hours into 4 time periods
        if self.hour < 6:
            time_bucket = "night"
        elif self.hour < 12:
            time_bucket = "morning"
        elif self.hour < 18:
            time_bucket = "afternoon"
        else:
            time_bucket = "evening"

        return f"{self.device}:{self.geo}:{time_bucket}"

    def to_dict(self) -> dict:
        return {"device": self.device, "geo": self.geo, "hour": self.hour}

    @classmethod
    def from_dict(cls, data: dict) -> "Context":
        return cls(
            device=data.get("device", "desktop"),
            geo=data.get("geo", "us"),
            hour=data.get("hour", 12)
        )

    @classmethod
    def now(cls, device: str = "desktop", geo: str = "us") -> "Context":
        """Create context with current hour."""
        return cls(device=device, geo=geo, hour=datetime.now().hour)


@dataclass
class ContextualArm:
    """Arm with per-context statistics."""
    page_id: str
    # Global stats (fallback when context has no data)
    global_alpha: int = 1
    global_beta: int = 1
    # Per-context stats: bucket -> (alpha, beta)
    context_stats: Dict[str, Tuple[int, int]] = field(default_factory=dict)

    def get_stats(self, context: Optional[Context] = None) -> Tuple[int, int]:
        """Get alpha/beta for a context, with hierarchical fallback."""
        if context is None:
            return self.global_alpha, self.global_beta

        bucket = context.to_bucket()

        # Try exact match
        if bucket in self.context_stats:
            return self.context_stats[bucket]

        # Try device:geo (ignore time)
        device_geo = f"{context.device}:{context.geo}"
        for key, stats in self.context_stats.items():
            if key.startswith(device_geo):
                return stats

        # Try device only
        for key, stats in self.context_stats.items():
            if key.startswith(f"{context.device}:"):
                return stats

        # Fallback to global
        return self.global_alpha, self.global_beta

    def sample(self, context: Optional[Context] = None) -> float:
        """Sample from posterior for given context."""
        alpha, beta = self.get_stats(context)
        return np.random.beta(alpha, beta)

    def update(self, converted: bool, context: Optional[Context] = None):
        """Update stats for context."""
        # Always update global
        if converted:
            self.global_alpha += 1
        else:
            self.global_beta += 1

        # Update context-specific if provided
        if context:
            bucket = context.to_bucket()
            alpha, beta = self.context_stats.get(bucket, (1, 1))
            if converted:
                self.context_stats[bucket] = (alpha + 1, beta)
            else:
                self.context_stats[bucket] = (alpha, beta + 1)

    @property
    def total(self) -> int:
        return (self.global_alpha - 1) + (self.global_beta - 1)

    @property
    def conversions(self) -> int:
        return self.global_alpha - 1

    @property
    def mean(self) -> float:
        return self.global_alpha / (self.global_alpha + self.global_beta)

    def mean_for_context(self, context: Context) -> float:
        alpha, beta = self.get_stats(context)
        return alpha / (alpha + beta)

    def to_dict(self) -> dict:
        return {
            "page_id": self.page_id,
            "global_alpha": self.global_alpha,
            "global_beta": self.global_beta,
            "context_stats": {k: list(v) for k, v in self.context_stats.items()}
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContextualArm":
        arm = cls(
            page_id=data["page_id"],
            global_alpha=data.get("global_alpha", 1),
            global_beta=data.get("global_beta", 1)
        )
        arm.context_stats = {
            k: tuple(v) for k, v in data.get("context_stats", {}).items()
        }
        return arm


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


class ContextualBandit:
    """
    Contextual Thompson Sampling Bandit.

    Learns optimal arm for each context (device, geo, time).
    Falls back hierarchically: exact context -> device+geo -> device -> global
    """

    def __init__(
        self,
        site_id: str,
        first_100_threshold: int = 100,
        neural_threshold: int = 1000,
        confidence_threshold: float = 0.95
    ):
        self.site_id = site_id
        self.arms: List[ContextualArm] = []
        self.first_100_threshold = first_100_threshold
        self.neural_threshold = neural_threshold
        self.confidence_threshold = confidence_threshold
        # Track context distribution for stats
        self.context_counts: Dict[str, int] = {}

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
        self.arms.append(ContextualArm(page_id=page_id))

    def select_arm(self, context: Optional[Context] = None) -> Tuple[int, str]:
        """Select best arm for the given context using Thompson Sampling."""
        if not self.arms:
            raise ValueError("No arms available")
        if len(self.arms) == 1:
            return 0, self.arms[0].page_id

        # Track context distribution
        if context:
            bucket = context.to_bucket()
            self.context_counts[bucket] = self.context_counts.get(bucket, 0) + 1

        # Sample from each arm's posterior for this context
        samples = [arm.sample(context) for arm in self.arms]
        best_idx = int(np.argmax(samples))
        return best_idx, self.arms[best_idx].page_id

    def update(self, page_id: str, converted: bool, context: Optional[Context] = None):
        """Update arm stats with outcome."""
        for arm in self.arms:
            if arm.page_id == page_id:
                arm.update(converted, context)
                return
        raise ValueError(f"Unknown page_id: {page_id}")

    def get_winner(self, context: Optional[Context] = None) -> Optional[Tuple[str, float]]:
        """Determine if there's a clear winner for the given context."""
        if len(self.arms) < 2:
            return None

        # Need minimum samples
        for arm in self.arms:
            alpha, beta = arm.get_stats(context)
            if (alpha - 1) + (beta - 1) < 3:
                return None

        n_samples = 10000
        samples = np.array([
            np.random.beta(*arm.get_stats(context), size=n_samples)
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

    def get_loser(self, context: Optional[Context] = None) -> Optional[str]:
        """Get the losing arm for a context."""
        winner = self.get_winner(context)
        if winner is None:
            return None
        winner_id, _ = winner
        for arm in self.arms:
            if arm.page_id != winner_id:
                return arm.page_id
        return None

    def get_divergence(self, context: Optional[Context] = None) -> float:
        """Get conversion rate divergence between arms."""
        if len(self.arms) < 2:
            return 0.0
        rates = [arm.mean_for_context(context) if context else arm.mean
                 for arm in self.arms]
        return max(rates) - min(rates)

    def get_stats(self, context: Optional[Context] = None) -> dict:
        """Get comprehensive stats, optionally filtered by context."""
        stats = {
            "site_id": self.site_id,
            "regime": self.regime,
            "total_sessions": self.total_sessions,
            "is_contextual": True,
            "arms": [
                {
                    "page_id": arm.page_id,
                    "sessions": arm.total,
                    "conversions": arm.conversions,
                    "global_rate": f"{arm.mean:.2%}",
                    "context_rate": f"{arm.mean_for_context(context):.2%}" if context else None,
                    "contexts_tracked": len(arm.context_stats)
                }
                for arm in self.arms
            ],
            "divergence": self.get_divergence(context),
            "winner": self.get_winner(context),
            "top_contexts": sorted(
                self.context_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:10]
        }

        if context:
            stats["query_context"] = context.to_dict()

        return stats

    def get_context_breakdown(self) -> dict:
        """Get per-context performance breakdown."""
        breakdown = {}

        # Collect all unique contexts
        all_contexts = set()
        for arm in self.arms:
            all_contexts.update(arm.context_stats.keys())

        for bucket in all_contexts:
            arm_stats = []
            for arm in self.arms:
                if bucket in arm.context_stats:
                    alpha, beta = arm.context_stats[bucket]
                    arm_stats.append({
                        "page_id": arm.page_id,
                        "sessions": (alpha - 1) + (beta - 1),
                        "rate": alpha / (alpha + beta)
                    })
            if arm_stats:
                breakdown[bucket] = {
                    "arms": arm_stats,
                    "traffic": self.context_counts.get(bucket, 0)
                }

        return breakdown

    def to_dict(self) -> dict:
        return {
            "site_id": self.site_id,
            "type": "contextual",
            "arms": [arm.to_dict() for arm in self.arms],
            "context_counts": self.context_counts,
            "first_100_threshold": self.first_100_threshold,
            "neural_threshold": self.neural_threshold,
            "confidence_threshold": self.confidence_threshold
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ContextualBandit":
        bandit = cls(
            site_id=data["site_id"],
            first_100_threshold=data.get("first_100_threshold", 100),
            neural_threshold=data.get("neural_threshold", 1000),
            confidence_threshold=data.get("confidence_threshold", 0.95)
        )
        bandit.arms = [ContextualArm.from_dict(a) for a in data.get("arms", [])]
        bandit.context_counts = data.get("context_counts", {})
        return bandit

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> "ContextualBandit":
        return cls.from_dict(json.loads(json_str))
