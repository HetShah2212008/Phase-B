"""
Email Agent — delivers the formatted report to a recipient via SMTP.

Pipeline position:
  RAG Agent → Content Agent → [Email Agent]

Responsibility:
  1. Read formatted_content and email_recipient from state
  2. Call EmailService.send_email()
  3. Write result to state["email_status"]: "sent" | "skipped" | "failed"

Skips gracefully when:
  - email_recipient is not set (pipeline run without email delivery)
  - formatted_content is empty (content agent produced nothing)
  - SMTP is not configured in .env

This means the full LangGraph pipeline can always run end-to-end even when
email delivery is not needed — just omit email_recipient from the initial state.
"""

import logging

from schemas.pipeline_state import PipelineState
from services.email_service import EmailService

logger = logging.getLogger(__name__)

_DEFAULT_SUBJECT = "AI Pipeline Report"


class EmailAgent:
    """
    Sends pipeline output via email when recipient and content are present.

    LangGraph node: last step in the Part 1 graph.
    """

    def __init__(self, email_service: EmailService | None = None) -> None:
        self._email = email_service or EmailService()

    async def run(self, state: PipelineState) -> PipelineState:
        """
        Send formatted_content to email_recipient.

        Sets state["email_status"] to one of:
          "sent"    — email delivered successfully
          "skipped" — no recipient or no content; nothing to send
          "failed"  — SMTP error occurred (logged, not re-raised)
        """
        recipient = (state.get("email_recipient") or "").strip()
        content   = (state.get("formatted_content") or "").strip()

        # --- Skip conditions ---
        if not recipient:
            logger.info("[EmailAgent] no email_recipient set — skipping delivery")
            state["email_status"] = "skipped"
            return state

        if not content:
            logger.warning(
                "[EmailAgent] formatted_content is empty — skipping delivery to %s",
                recipient,
            )
            state["email_status"] = "skipped"
            return state

        if not self._email.is_configured():
            logger.warning(
                "[EmailAgent] SMTP not configured — skipping delivery to %s", recipient
            )
            state["email_status"] = "skipped"
            return state

        # --- Send ---
        try:
            result = await self._email.send_email(
                to_email=recipient,
                subject=_DEFAULT_SUBJECT,
                body=content,
            )
            state["email_status"] = result.get("status", "sent")
            logger.info(
                "[EmailAgent] email delivered to %s — status=%s",
                recipient, state["email_status"],
            )

        except Exception as exc:  # noqa: BLE001
            # Email failure is non-fatal — the report was already generated.
            # Log the error and mark status so callers can inspect it.
            logger.error(
                "[EmailAgent] send failed to %s (%s: %s)",
                recipient, type(exc).__name__, exc,
            )
            state["email_status"] = "failed"

        return state
