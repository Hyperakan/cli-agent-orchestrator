"""Unit tests for Feynman provider."""

from unittest.mock import MagicMock, patch

import pytest

from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.providers.feynman import FeynmanProvider, ProviderError

FEYNMAN_IDLE_OUTPUT = """
Feynman Research Agent v0.3.48
Model: gemini-2.5-pro

researcher ❯
"""

FEYNMAN_IDLE_CUSTOM_SYMBOL_OUTPUT = """
Feynman Research Agent online.
✦ Loaded 15 research tools.

researcher ✦
"""

FEYNMAN_PROCESSING_OUTPUT = """
● Run a literature review on quantum computing

Searching arXiv...
Thinking...
⏱ 12s
"""

FEYNMAN_WAITING_APPROVAL_OUTPUT = """
● Run a dangerous experiment command

  ⚠️  DANGEROUS COMMAND: execute local script
      rm -rf ./tmp

      [o]nce  |  [s]ession  |  [a]lways  |  [d]eny

      Choice [o/s/a/D]:
"""

FEYNMAN_COMPLETED_OUTPUT = """
● Summarize this paper

─ Feynman Assistant ─
The paper introduces a novel transformer architecture for protein design.

researcher ❯
"""

FEYNMAN_COMPLETED_CUSTOM_OUTPUT = """
● Analyze findings

The experimental validation confirms the hypothesized binding affinity.

researcher ❯
"""

FEYNMAN_NO_RESPONSE_OUTPUT = """
● Tell me about this paper
"""


class TestFeynmanCommandBuilding:
    def _profile(self, **kwargs):
        profile = MagicMock()
        profile.model = kwargs.get("model", None)
        profile.feynmanConfig = kwargs.get("feynmanConfig", None)
        return profile

    @patch("cli_agent_orchestrator.providers.feynman.load_agent_profile")
    def test_build_command_default(self, mock_load):
        mock_load.return_value = self._profile()
        provider = FeynmanProvider("tid", "sess", "win", "academic_researcher")
        assert provider._build_feynman_command() == "feynman"

    @patch("cli_agent_orchestrator.providers.feynman.load_agent_profile")
    def test_build_command_with_profile_model(self, mock_load):
        mock_load.return_value = self._profile(model="gemini-2.5-pro")
        provider = FeynmanProvider("tid", "sess", "win", "academic_researcher")
        assert provider._build_feynman_command() == "feynman --model gemini-2.5-pro"

    @patch("cli_agent_orchestrator.providers.feynman.load_agent_profile")
    def test_build_command_with_per_call_model_override(self, mock_load):
        mock_load.return_value = self._profile(model="gemini-2.5-pro")
        provider = FeynmanProvider("tid", "sess", "win", "academic_researcher", model="claude-3-7-sonnet")
        assert provider._build_feynman_command() == "feynman --model claude-3-7-sonnet"

    @patch("cli_agent_orchestrator.providers.feynman.load_agent_profile")
    def test_build_command_with_thinking_config(self, mock_load):
        mock_load.return_value = self._profile(
            model="gemini-2.5-pro",
            feynmanConfig={"thinking": "high"},
        )
        provider = FeynmanProvider("tid", "sess", "win", "academic_researcher")
        assert provider._build_feynman_command() == "feynman --model gemini-2.5-pro --thinking high"

    @patch("cli_agent_orchestrator.providers.feynman.load_agent_profile")
    def test_build_command_raises_on_invalid_profile(self, mock_load):
        mock_load.side_effect = ValueError("Profile not found")
        provider = FeynmanProvider("tid", "sess", "win", "missing_profile")
        with pytest.raises(ProviderError, match="Failed to load agent profile"):
            provider._build_feynman_command()


class TestFeynmanStatusDetection:
    def test_get_status_idle(self):
        provider = FeynmanProvider("tid", "sess", "win", None)
        assert provider.get_status(FEYNMAN_IDLE_OUTPUT) == TerminalStatus.IDLE

    def test_get_status_idle_custom_symbol(self):
        provider = FeynmanProvider("tid", "sess", "win", None)
        assert provider.get_status(FEYNMAN_IDLE_CUSTOM_SYMBOL_OUTPUT) == TerminalStatus.IDLE

    def test_get_status_processing(self):
        provider = FeynmanProvider("tid", "sess", "win", None)
        assert provider.get_status(FEYNMAN_PROCESSING_OUTPUT) == TerminalStatus.PROCESSING

    def test_get_status_waiting_user_answer(self):
        provider = FeynmanProvider("tid", "sess", "win", None)
        assert provider.get_status(FEYNMAN_WAITING_APPROVAL_OUTPUT) == TerminalStatus.WAITING_USER_ANSWER

    def test_get_status_completed(self):
        provider = FeynmanProvider("tid", "sess", "win", None)
        assert provider.get_status(FEYNMAN_COMPLETED_OUTPUT) == TerminalStatus.COMPLETED

    def test_get_status_completed_custom(self):
        provider = FeynmanProvider("tid", "sess", "win", None)
        assert provider.get_status(FEYNMAN_COMPLETED_CUSTOM_OUTPUT) == TerminalStatus.COMPLETED

    def test_get_status_error(self):
        provider = FeynmanProvider("tid", "sess", "win", None)
        assert provider.get_status("Error: model execution failed\n") == TerminalStatus.ERROR

    def test_get_status_empty_output(self):
        provider = FeynmanProvider("tid", "sess", "win", None)
        assert provider.get_status("") == TerminalStatus.ERROR


class TestFeynmanExtraction:
    def test_extract_last_message_with_header(self):
        provider = FeynmanProvider("tid", "sess", "win", None)
        assert (
            provider.extract_last_message_from_script(FEYNMAN_COMPLETED_OUTPUT)
            == "The paper introduces a novel transformer architecture for protein design."
        )

    def test_extract_last_message_without_header(self):
        provider = FeynmanProvider("tid", "sess", "win", None)
        assert (
            provider.extract_last_message_from_script(FEYNMAN_COMPLETED_CUSTOM_OUTPUT)
            == "The experimental validation confirms the hypothesized binding affinity."
        )

    def test_extract_last_message_missing_response_raises(self):
        provider = FeynmanProvider("tid", "sess", "win", None)
        with pytest.raises(ValueError, match="Empty Feynman response|No Feynman response found"):
            provider.extract_last_message_from_script(FEYNMAN_IDLE_OUTPUT)

    def test_extract_last_message_empty_raises(self):
        provider = FeynmanProvider("tid", "sess", "win", None)
        with pytest.raises(ValueError, match="No Feynman response found|Empty Feynman response"):
            provider.extract_last_message_from_script(FEYNMAN_NO_RESPONSE_OUTPUT)

    def test_exit_cli_command(self):
        provider = FeynmanProvider("tid", "sess", "win", None)
        assert provider.exit_cli() == "/exit"
