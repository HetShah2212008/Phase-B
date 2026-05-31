"""
Email service — delivers reports via SMTP.

This module provides two interfaces:

1. Module-level async function (used by financial_pipeline.py):
       success = await send_email_report(report_content, recipient_email)

2. Class-based service (used by EmailAgent / langgraph_pipeline.py):
       await EmailService().send_email(to, subject, body)

Credentials are read from environment variables at call time so the module
can be imported without any configuration being present yet.

Required environment variables:
    SMTP_SENDER_EMAIL     — the Gmail address used to send
    SMTP_SENDER_PASSWORD  — Gmail App Password (not your account password)
    SMTP_SERVER           — defaults to smtp.gmail.com
    SMTP_PORT             — defaults to 587

Gmail setup reminder:
    1. Enable 2-Step Verification on your Google account.
    2. Go to Google Account → Security → App Passwords.
    3. Generate a password for "Mail" and paste it as SMTP_SENDER_PASSWORD.
"""

import logging
import os
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)

_DEFAULT_SUBJECT = "AI Pipeline Report"


# ---------------------------------------------------------------------------
# Module-level async function — primary interface for financial_pipeline.py
# ---------------------------------------------------------------------------

async def send_email_report(report_content: str, recipient_email: str) -> bool:
    """
    Send a formatted report to a recipient via Gmail SMTP.

    Reads credentials dynamically from environment variables so no
    configuration object is needed — works standalone or inside FastAPI.

    Args:
        report_content:  The Markdown/plain-text report body to send.
        recipient_email: Destination email address.

    Returns:
        True  — email was delivered successfully.
        False — delivery failed (error is logged, not re-raised).
    """
    if not recipient_email.strip():
        logger.error("[send_email_report] recipient_email is empty — aborting")
        return False

    if not report_content.strip():
        logger.error("[send_email_report] report_content is empty — aborting")
        return False

    # Read credentials at call time so .env changes take effect without restart
    sender_email = os.getenv("SMTP_SENDER_EMAIL", "")
    sender_password = os.getenv("SMTP_SENDER_PASSWORD", "")
    smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("SMTP_PORT", "587"))

    if not sender_email or not sender_password:
        logger.error(
            "[send_email_report] SMTP_SENDER_EMAIL or SMTP_SENDER_PASSWORD not set — "
            "configure them in .env to enable email delivery"
        )
        return False

    msg = MIMEMultipart()
    msg["Subject"] = _DEFAULT_SUBJECT
    msg["From"]    = sender_email
    msg["To"]      = recipient_email
    msg.attach(MIMEText(report_content, "plain", "utf-8"))

    try:
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, [recipient_email], msg.as_string())

        logger.info(
            "[send_email_report] delivered to %s via %s:%d",
            recipient_email, smtp_server, smtp_port,
        )
        return True

    except smtplib.SMTPAuthenticationError:
        logger.error(
            "[send_email_report] authentication failed — "
            "check SMTP_SENDER_EMAIL and SMTP_SENDER_PASSWORD"
        )
        return False

    except smtplib.SMTPException as exc:
        logger.error("[send_email_report] SMTP error: %s", exc)
        return False

    except Exception as exc:  # noqa: BLE001
        logger.error("[send_email_report] unexpected error: %s", exc)
        return False


# ---------------------------------------------------------------------------
# Class-based service — used by EmailAgent / langgraph_pipeline.py
# ---------------------------------------------------------------------------

class EmailService:
    """
    Sends plain-text or HTML emails via SMTP (Gmail-compatible).

    Reads credentials from config (SMTP_USER, SMTP_PASSWORD, etc.) via
    pydantic-settings so it integrates cleanly with the FastAPI app.
    """

    def __init__(self) -> None:
        # Import here to avoid circular imports when used standalone
        from config import get_settings
        settings = get_settings()
        self._host       = settings.smtp_host
        self._port       = settings.smtp_port
        self._user       = settings.smtp_user
        self._password   = settings.smtp_password
        self._from_email = settings.smtp_from_email or settings.smtp_user

    def is_configured(self) -> bool:
        """True when minimum SMTP settings are present."""
        return bool(self._user and self._password and self._from_email)

    async def send_email(
        self,
        to_email: str,
        subject: str,
        body: str,
        *,
        html: bool = False,
    ) -> dict[str, str]:
        """
        Send an email. Returns a status dict for pipeline state.

        Raises:
            ValueError: Missing fields or SMTP not configured.
            RuntimeError: SMTP connection or send failure.
        """
        if not to_email.strip():
            raise ValueError("Recipient email (to_email) is required.")
        if not subject.strip():
            raise ValueError("Email subject is required.")
        if not body.strip():
            raise ValueError("Email body is required.")
        if not self.is_configured():
            raise ValueError(
                "SMTP is not configured. Set SMTP_USER, SMTP_PASSWORD, and "
                "SMTP_FROM_EMAIL in .env."
            )

        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = self._from_email
        msg["To"]      = to_email
        msg.attach(MIMEText(body, "html" if html else "plain", "utf-8"))

        try:
            with smtplib.SMTP(self._host, self._port) as server:
                server.starttls()
                server.login(self._user, self._password)
                server.sendmail(self._from_email, [to_email], msg.as_string())

            logger.info("[EmailService] sent to %s", to_email)
            return {"status": "sent", "message": f"Email delivered to {to_email}"}

        except smtplib.SMTPException as exc:
            logger.exception("[EmailService] SMTP send failed")
            raise RuntimeError(f"Failed to send email: {exc}") from exc
