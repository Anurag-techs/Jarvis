"""
JARVIS Planner Persistence Repository.
Provides clean saving, loading, and listing of active and completed plans using Pydantic JSON files.
"""

import json
import logging
import os
from typing import Optional

from backend.services.planner.models import Plan

logger = logging.getLogger("jarvis.services.planner.persistence")


class PlanRepository:
    """Handles plan serialization and deserialization to the local filesystem."""

    def __init__(self, storage_dir: str = "instance/plans") -> None:
        """Initialize repository.

        Args:
            storage_dir: Absolute or relative workspace path to save plans.
        """
        self._storage_dir = storage_dir
        os.makedirs(self._storage_dir, exist_ok=True)

    def save_plan(self, plan: Plan) -> None:
        """Saves a plan state to file as JSON.

        Args:
            plan: The Plan instance to persist.
        """
        filepath = os.path.join(self._storage_dir, f"{plan.id}.json")
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(plan.model_dump_json(indent=2))
            logger.debug("Successfully saved plan %s to %s", plan.id, filepath)
        except Exception as exc:
            logger.error("Failed to persist plan %s: %s", plan.id, exc)

    def load_plan(self, plan_id: str) -> Optional[Plan]:
        """Loads a plan state from file.

        Args:
            plan_id: Target plan identifier.

        Returns:
            Plan instance if found and parsed successfully, None otherwise.
        """
        filepath = os.path.join(self._storage_dir, f"{plan_id}.json")
        if not os.path.exists(filepath):
            logger.debug("Plan file %s not found.", filepath)
            return None

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                data = json.load(f)
            return Plan.model_validate(data)
        except Exception as exc:
            logger.error("Failed to load or parse plan %s: %s", plan_id, exc)
            return None

    def list_plans(self) -> list[Plan]:
        """Lists all plans persisted in the storage directory.

        Returns:
            List of Plan instances.
        """
        plans = []
        try:
            for filename in os.listdir(self._storage_dir):
                if filename.endswith(".json"):
                    plan_id = filename.replace(".json", "")
                    plan = self.load_plan(plan_id)
                    if plan:
                        plans.append(plan)
        except Exception as exc:
            logger.error("Failed to list plans directory: %s", exc)
        return plans
