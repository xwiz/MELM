"""Tests for assistant_skill_mailbox (Hook #3).

Covers: MailboxConfig.from_env, is_configured, fetch_inbox, send_message,
intent helpers (is_send_intent, is_read_intent, extract_mail_body,
resolve_recipient), summarise_inbox, and router _mail() dispatch.

All network calls are bypassed via os.environ mocking — no live server needed.
"""

import os
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# MailboxConfig
# ---------------------------------------------------------------------------

class MailboxConfigTests(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in (
            "MELM_MAILBOX_EMAIL", "MELM_MAILBOX_PASSWORD",
            "MELM_MAILBOX_IMAP_HOST", "MELM_MAILBOX_SMTP_HOST",
            "MELM_MAILBOX_USERNAME", "MELM_MAILBOX_IMAP_PORT",
            "MELM_MAILBOX_SMTP_PORT",
        )}

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def _set_env(self, **kwargs):
        defaults = dict(
            MELM_MAILBOX_EMAIL="device@venia.cloud",
            MELM_MAILBOX_PASSWORD="s3cr3t",
            MELM_MAILBOX_IMAP_HOST="imap.venia.cloud",
            MELM_MAILBOX_SMTP_HOST="smtp.venia.cloud",
        )
        defaults.update(kwargs)
        os.environ.update(defaults)

    def test_from_env_returns_none_when_missing(self):
        from melm.appliance.assistant_skill_mailbox import MailboxConfig
        self.assertIsNone(MailboxConfig.from_env())

    def test_from_env_returns_config_when_full(self):
        from melm.appliance.assistant_skill_mailbox import MailboxConfig
        self._set_env()
        cfg = MailboxConfig.from_env()
        self.assertIsNotNone(cfg)
        self.assertEqual(cfg.email, "device@venia.cloud")
        self.assertEqual(cfg.imap_port, 993)
        self.assertEqual(cfg.smtp_port, 587)

    def test_from_env_uses_username_override(self):
        from melm.appliance.assistant_skill_mailbox import MailboxConfig
        self._set_env(MELM_MAILBOX_USERNAME="device_user")
        cfg = MailboxConfig.from_env()
        self.assertEqual(cfg.username, "device_user")

    def test_from_env_username_defaults_to_email(self):
        from melm.appliance.assistant_skill_mailbox import MailboxConfig
        self._set_env()
        cfg = MailboxConfig.from_env()
        self.assertEqual(cfg.username, cfg.email)

    def test_from_env_custom_ports(self):
        from melm.appliance.assistant_skill_mailbox import MailboxConfig
        self._set_env(MELM_MAILBOX_IMAP_PORT="143", MELM_MAILBOX_SMTP_PORT="465")
        cfg = MailboxConfig.from_env()
        self.assertEqual(cfg.imap_port, 143)
        self.assertEqual(cfg.smtp_port, 465)

    def test_is_configured_false_when_missing(self):
        from melm.appliance.assistant_skill_mailbox import is_configured
        self.assertFalse(is_configured())

    def test_is_configured_true_when_full(self):
        from melm.appliance.assistant_skill_mailbox import is_configured
        self._set_env()
        self.assertTrue(is_configured())


# ---------------------------------------------------------------------------
# fetch_inbox (mocked IMAP)
# ---------------------------------------------------------------------------

class FetchInboxTests(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in (
            "MELM_MAILBOX_EMAIL", "MELM_MAILBOX_PASSWORD",
            "MELM_MAILBOX_IMAP_HOST", "MELM_MAILBOX_SMTP_HOST",
        )}

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_returns_error_string_when_not_configured(self):
        from melm.appliance.assistant_skill_mailbox import fetch_inbox
        result = fetch_inbox()
        self.assertIsInstance(result, str)
        self.assertIn("not configured", result)

    def test_returns_error_string_on_imap_failure(self):
        os.environ.update({
            "MELM_MAILBOX_EMAIL": "x@v.cloud",
            "MELM_MAILBOX_PASSWORD": "pw",
            "MELM_MAILBOX_IMAP_HOST": "imap.v.cloud",
            "MELM_MAILBOX_SMTP_HOST": "smtp.v.cloud",
        })
        with patch("imaplib.IMAP4_SSL") as mock_imap:
            mock_imap.side_effect = OSError("connection refused")
            from melm.appliance.assistant_skill_mailbox import fetch_inbox
            result = fetch_inbox()
        self.assertIsInstance(result, str)

    def test_returns_summaries_on_success(self):
        os.environ.update({
            "MELM_MAILBOX_EMAIL": "x@v.cloud",
            "MELM_MAILBOX_PASSWORD": "pw",
            "MELM_MAILBOX_IMAP_HOST": "imap.v.cloud",
            "MELM_MAILBOX_SMTP_HOST": "smtp.v.cloud",
        })
        raw_header = (
            b"From: sender@example.com\r\n"
            b"Subject: Hello world\r\n"
            b"Date: Mon, 01 Jan 2026 00:00:00 +0000\r\n\r\n"
        )
        mock_imap = MagicMock()
        mock_imap.__enter__ = lambda s: s
        mock_imap.__exit__ = MagicMock(return_value=False)
        mock_imap.search.return_value = (None, [b"1"])
        mock_imap.fetch.return_value = (None, [(b"1 (RFC822.HEADER {92})", raw_header)])
        mock_imap.socket.return_value = MagicMock()
        mock_imap.login.return_value = ("OK", [])
        mock_imap.select.return_value = ("OK", [b"1"])

        with patch("imaplib.IMAP4_SSL", return_value=mock_imap):
            from melm.appliance import assistant_skill_mailbox as m
            result = m.fetch_inbox()
        self.assertIsInstance(result, list)
        self.assertEqual(result[0].sender, "sender@example.com")
        self.assertEqual(result[0].subject, "Hello world")


# ---------------------------------------------------------------------------
# send_message (mocked SMTP)
# ---------------------------------------------------------------------------

class SendMessageTests(unittest.TestCase):
    def setUp(self):
        self._saved = {k: os.environ.pop(k, None) for k in (
            "MELM_MAILBOX_EMAIL", "MELM_MAILBOX_PASSWORD",
            "MELM_MAILBOX_IMAP_HOST", "MELM_MAILBOX_SMTP_HOST",
        )}

    def tearDown(self):
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_returns_error_when_not_configured(self):
        from melm.appliance.assistant_skill_mailbox import send_message
        result = send_message("to@x.com", "hi", "body")
        self.assertIsInstance(result, str)

    def test_returns_true_on_success(self):
        os.environ.update({
            "MELM_MAILBOX_EMAIL": "x@v.cloud",
            "MELM_MAILBOX_PASSWORD": "pw",
            "MELM_MAILBOX_IMAP_HOST": "imap.v.cloud",
            "MELM_MAILBOX_SMTP_HOST": "smtp.v.cloud",
        })
        mock_smtp = MagicMock()
        mock_smtp.__enter__ = lambda s: s
        mock_smtp.__exit__ = MagicMock(return_value=False)
        with patch("smtplib.SMTP", return_value=mock_smtp):
            from melm.appliance import assistant_skill_mailbox as m
            result = m.send_message("to@x.com", "hi", "body text")
        self.assertIs(result, True)

    def test_returns_error_string_on_smtp_failure(self):
        os.environ.update({
            "MELM_MAILBOX_EMAIL": "x@v.cloud",
            "MELM_MAILBOX_PASSWORD": "pw",
            "MELM_MAILBOX_IMAP_HOST": "imap.v.cloud",
            "MELM_MAILBOX_SMTP_HOST": "smtp.v.cloud",
        })
        with patch("smtplib.SMTP") as mock_cls:
            mock_cls.side_effect = OSError("refused")
            from melm.appliance import assistant_skill_mailbox as m
            result = m.send_message("to@x.com", "hi", "body")
        self.assertIsInstance(result, str)


# ---------------------------------------------------------------------------
# Intent helpers
# ---------------------------------------------------------------------------

class IntentHelperTests(unittest.TestCase):
    def setUp(self):
        from melm.appliance.assistant_skill_mailbox import (
            extract_mail_body,
            is_read_intent,
            is_send_intent,
            resolve_recipient,
        )
        self.send = is_send_intent
        self.read = is_read_intent
        self.body = extract_mail_body
        self.recipient = resolve_recipient

    def test_send_intent_email_to(self):
        self.assertTrue(self.send(("send", "email", "to", "mom")))

    def test_send_intent_write_email_to(self):
        self.assertTrue(self.send(("write", "email", "to", "leo")))

    def test_read_intent_check_email(self):
        self.assertTrue(self.read(("check", "my", "email")))

    def test_read_intent_show_inbox(self):
        self.assertTrue(self.read(("show", "inbox")))

    def test_send_not_triggered_without_email(self):
        self.assertFalse(self.send(("call", "mom")))

    def test_read_not_triggered_without_email_token(self):
        self.assertFalse(self.read(("read", "a", "story")))

    def test_extract_body_after_that(self):
        self.assertEqual(self.body("email mom that i am home"), "i am home")

    def test_extract_body_after_saying(self):
        self.assertEqual(self.body("send email to dad saying i will be late"), "i will be late")

    def test_extract_body_empty_when_no_marker(self):
        self.assertEqual(self.body("check email"), "")

    def test_resolve_recipient_from_contacts(self):
        contacts = {"mom": "mom@venia.cloud", "leo": "leo@venia.cloud"}
        result = self.recipient(("email", "mom", "that", "i", "am", "home"), contacts)
        self.assertEqual(result, ("mom", "mom@venia.cloud"))

    def test_resolve_recipient_none_when_no_match(self):
        self.assertIsNone(self.recipient(("email", "friend"), {}))

    def test_resolve_recipient_inline_address(self):
        result = self.recipient(("email", "dad@example.com"), {})
        self.assertEqual(result, ("dad@example.com", "dad@example.com"))


# ---------------------------------------------------------------------------
# summarise_inbox
# ---------------------------------------------------------------------------

class SummariseInboxTests(unittest.TestCase):
    def setUp(self):
        from melm.appliance.assistant_skill_mailbox import MailSummary, summarise_inbox
        self.summarise = summarise_inbox
        self.MailSummary = MailSummary

    def test_empty_inbox(self):
        result = self.summarise([])
        self.assertIn("empty", result)

    def test_single_message(self):
        s = self.MailSummary("1", "a@b.com", "Hello", "Mon")
        result = self.summarise([s])
        self.assertIn("1 unread", result)
        self.assertIn("a@b.com", result)

    def test_truncates_at_five(self):
        msgs = [self.MailSummary(str(i), f"s{i}@b.com", f"Subj {i}", "") for i in range(8)]
        result = self.summarise(msgs)
        self.assertIn("3 more", result)


# ---------------------------------------------------------------------------
# Router _mail() dispatch
# ---------------------------------------------------------------------------

class MailRouterDispatchTests(unittest.TestCase):
    def _make_router(self, contacts=None):
        from melm.appliance.local_assistant_router import (
            LocalAssistantProfile,
            OnDeviceAssistantRouter,
        )
        profile = LocalAssistantProfile(
            contacts=contacts or {"mom": "mom@venia.cloud", "leo": "leo@venia.cloud"},
        )
        return OnDeviceAssistantRouter(profile=profile)

    def _clear_mailbox_env(self):
        for k in ("MELM_MAILBOX_EMAIL", "MELM_MAILBOX_PASSWORD",
                  "MELM_MAILBOX_IMAP_HOST", "MELM_MAILBOX_SMTP_HOST"):
            os.environ.pop(k, None)

    def test_not_configured_returns_clarify(self):
        self._clear_mailbox_env()
        router = self._make_router()
        decision = router.handle("check my email")
        self.assertEqual(decision.intent, "mail")
        self.assertEqual(decision.route, "clarify")
        self.assertIn("provisioning", decision.answer.lower())

    def test_read_inbox_returns_local_answer(self):
        os.environ.update({
            "MELM_MAILBOX_EMAIL": "x@v.cloud",
            "MELM_MAILBOX_PASSWORD": "pw",
            "MELM_MAILBOX_IMAP_HOST": "imap.v.cloud",
            "MELM_MAILBOX_SMTP_HOST": "smtp.v.cloud",
        })
        with patch("melm.appliance.assistant_skill_mailbox.fetch_inbox", return_value=[]):
            router = self._make_router()
            decision = router.handle("check my email")
        self.assertEqual(decision.intent, "mail")
        self.assertEqual(decision.route, "local_answer")
        self.assertIn("empty", decision.answer)

    def test_send_missing_recipient_returns_clarify(self):
        os.environ.update({
            "MELM_MAILBOX_EMAIL": "x@v.cloud",
            "MELM_MAILBOX_PASSWORD": "pw",
            "MELM_MAILBOX_IMAP_HOST": "imap.v.cloud",
            "MELM_MAILBOX_SMTP_HOST": "smtp.v.cloud",
        })
        router = self._make_router(contacts={})
        decision = router.handle("send email to stranger")
        self.assertEqual(decision.intent, "mail")
        self.assertEqual(decision.route, "clarify")

    def test_send_to_contact_returns_device_action(self):
        os.environ.update({
            "MELM_MAILBOX_EMAIL": "x@v.cloud",
            "MELM_MAILBOX_PASSWORD": "pw",
            "MELM_MAILBOX_IMAP_HOST": "imap.v.cloud",
            "MELM_MAILBOX_SMTP_HOST": "smtp.v.cloud",
        })
        with patch("melm.appliance.assistant_skill_mailbox.send_message", return_value=True):
            router = self._make_router()
            decision = router.handle("send email to mom that i am home")
        self.assertEqual(decision.intent, "mail")
        self.assertEqual(decision.route, "device_action")
        self.assertTrue(decision.device_action)


if __name__ == "__main__":
    unittest.main()
