"""
JARVIS User Approval Service Abstractions.
Defines contracts and concrete providers to verify explicit user intent for desktop actions.
"""

from abc import ABC, abstractmethod
import logging

logger = logging.getLogger("jarvis.services.approval")


class BaseApprovalService(ABC):
    """Abstract Base Class for confirming explicit user intent before executing desktop actions."""

    @abstractmethod
    def request_approval(self, action_description: str) -> bool:
        """Prompts the user for permission to execute a specific action.

        Args:
            action_description: A human-readable summary of the action.

        Returns:
            True if approved, False if declined.
        """


class TerminalApprovalService(BaseApprovalService):
    """Interactive command-line approval service using standard input."""

    def request_approval(self, action_description: str) -> bool:
        print(f"\n⚠️  [USER APPROVAL REQUIRED]: {action_description}")
        try:
            response = input("Do you want to allow JARVIS to perform this action? (y/N): ").strip().lower()
            return response in ("y", "yes")
        except (KeyboardInterrupt, EOFError):
            print("\nApproval declined (interrupted).")
            return False


class AutoApprovalService(BaseApprovalService):
    """Automated approval service for testing and headless environments."""

    def __init__(self, approved: bool = True) -> None:
        self.approved = approved

    def request_approval(self, action_description: str) -> bool:
        logger.info("Auto-approval resolved (%s) for action: %s", self.approved, action_description)
        return self.approved
