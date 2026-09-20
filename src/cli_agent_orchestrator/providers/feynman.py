"""Feynman Agent provider implementation."""

import logging
import os
import re
import shlex
from typing import Optional

from cli_agent_orchestrator.backends.registry import get_backend
from cli_agent_orchestrator.models.terminal import TerminalStatus
from cli_agent_orchestrator.providers.base import BaseProvider
from cli_agent_orchestrator.services.settings_service import get_server_settings
from cli_agent_orchestrator.utils.agent_profiles import load_agent_profile
from cli_agent_orchestrator.utils.terminal import wait_for_shell, wait_until_status

logger = logging.getLogger(__name__)

ANSI_CODE_PATTERN = r"\x1b(?:\[[0-9;?]*[ -/]*[@-~]|\][^\x07]*(?:\x07|\x1b\\))"
IDLE_PROMPT_PATTERN = os.environ.get(
    "CAO_FEYNMAN_IDLE_PROMPT_REGEX",
    r"(?:^(?!.*(?:Thinking|musing|Searching|Fetching|Synthesizing|Auditing|Running|Ctrl\+C cancel|/exit)).{0,80}(?:❯|✦|>)\s*$|Type your message|•\s*(?:low|medium|high))",
)
IDLE_PROMPT_PATTERN_LOG = os.environ.get("CAO_FEYNMAN_IDLE_LOG_REGEX", r"(?:[❯✦>]|Type your message)")
PROCESSING_PATTERN = os.environ.get(
    "CAO_FEYNMAN_PROCESSING_REGEX",
    r"(?:Thinking|musing\.\.\.|Searching|Fetching|Synthesizing|Auditing|Running|Ctrl\+C cancel|⏱\s*\d+s)",
)
ACTIVE_PROCESSING_PATTERN = r"(?:Thinking|musing\.\.\.|Searching|Fetching|Synthesizing|Auditing|Running|Ctrl\+C cancel|⏱\s*\d+s)"
WAITING_PROMPT_PATTERN = (
    r"(?:Approve|Allow|Proceed|Confirm)[^\n]*(?:y/n|yes/no|\[y/N\])"
    r"|(?:DANGEROUS COMMAND|危险命令)"
    r"|(?:\[o\](?:nce|仅此一次).*\[s\](?:ession|本次会话).*\[d\](?:eny|拒绝))"
    r"|(?:(?:Choice|选择)\s+\[o/s(?:/a)?/D\]:)"
    r"|(?:Allow once.*Allow always.*Reject)"
    r"|(?:needs your input|Other \(type (?:your answer|below)\))"
    r"|(?:↑/↓\s*(?:to )?(?:select|navigate).*Enter)"
    r"|(?:type your answer and press Enter)"
)
SETUP_REQUIRED_PATTERN = r"(?:Feynman setup|Choose how to configure model access|No authenticated Pi models)"
ERROR_PATTERN = r"^(?:Error:|ERROR:|Traceback \(most recent call last\):|feynman .*failed:)"
USER_PREFIX_PATTERN = os.environ.get("CAO_FEYNMAN_USER_PREFIX_REGEX", r"^(?:●|>)\s+")
ASSISTANT_HEADER_PATTERN = os.environ.get(
    "CAO_FEYNMAN_ASSISTANT_HEADER_REGEX",
    r"─\s+.*(?:Feynman|Researcher|Verifier|Reviewer|Assistant).*\s+─",
)
STATUS_IDLE_TIMER_PATTERN = r"⏲\s*([^\s│]+)"
SEPARATOR_PATTERN = r"^[\s─━═-]{10,}$|^[\s─━═-]+\s+.*\s+[\s─━═-]+$"
STATUS_LINE_PATTERN = r"^.*(?:YOLO|ctx|⏲|⏱|msg=interrupt|Ctrl\+C cancel).*$"
MAX_STABLE_IDLE_TIMER_POLLS = int(os.environ.get("CAO_FEYNMAN_MAX_STABLE_IDLE_POLLS", "8"))


class ProviderError(Exception):
    """Exception raised for Feynman provider-specific errors."""

    pass


def _strip_ansi(text: str) -> str:
    return re.sub(ANSI_CODE_PATTERN, "", text)


def _is_idle_line(line: str) -> bool:
    return re.search(IDLE_PROMPT_PATTERN, line) is not None


def _is_chrome_line(line: str) -> bool:
    stripped = line.strip()
    return (
        not stripped
        or _is_idle_line(stripped)
        or re.match(SEPARATOR_PATTERN, stripped) is not None
        or re.match(STATUS_LINE_PATTERN, stripped) is not None
        or re.search(PROCESSING_PATTERN, stripped, re.IGNORECASE) is not None
        or re.search(ASSISTANT_HEADER_PATTERN, stripped) is not None
        or re.search(USER_PREFIX_PATTERN, stripped) is not None
    )


def _last_idle_timer(text: str) -> Optional[str]:
    matches = list(re.finditer(STATUS_IDLE_TIMER_PATTERN, text))
    return matches[-1].group(1) if matches else None


def _has_waiting_prompt(text: str) -> bool:
    """Detect active Feynman approval or clarify prompts near the prompt area."""
    if re.search(WAITING_PROMPT_PATTERN, text, re.IGNORECASE | re.MULTILINE):
        clarify_context = re.search(
            r"(?:needs your input|Other \(type (?:your answer|below)\)|"
            r"↑/↓\s*(?:to )?select.*Enter (?:to )?confirm|"
            r"type your answer and press Enter)",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        approval_context = re.search(
            r"(?:Choice|选择)\s+\[o/s(?:/a)?/D\]:|"
            r"\[o\](?:nce|仅此一次).*\[s\](?:ession|本次会话).*\[d\](?:eny|拒绝)|"
            r"Allow once.*Allow always.*Reject|"
            r"(?:Approve|Allow|Proceed|Confirm)[^\n]*(?:y/n|yes/no|\[y/N\])|"
            r"(?:DANGEROUS COMMAND|危险命令)",
            text,
            re.IGNORECASE | re.MULTILINE,
        )
        return bool(clarify_context or approval_context)
    return False


class FeynmanProvider(BaseProvider):
    """Provider for a CAO-managed Feynman AI Research Agent profile.

    Launches the Feynman interactive CLI in a tmux pane and bridges CAO
    orchestration commands with Feynman's research workflows.
    """

    def __init__(
        self,
        terminal_id: str,
        session_name: str,
        window_name: str,
        agent_profile: Optional[str] = None,
        allowed_tools: Optional[list] = None,
        skill_prompt: Optional[str] = None,
        model: Optional[str] = None,
    ):
        super().__init__(terminal_id, session_name, window_name, allowed_tools, skill_prompt)
        self._initialized = False
        self._agent_profile = agent_profile
        # Explicit per-call override for profile.model, see _build_feynman_command.
        self._model = model
        self._last_idle_timer: Optional[str] = None
        self._stable_idle_timer_count = 0

    @property
    def paste_enter_count(self) -> int:
        """Feynman submits bracketed paste with a single Enter."""
        return 1

    @property
    def blocks_orchestrated_input_while_waiting_user_answer(self) -> bool:
        """Feynman approval and clarify pickers consume pasted text as answers."""
        return True

    def _build_feynman_command(self) -> str:
        """Build the Feynman launch command from the CAO agent profile."""
        profile = None
        if self._agent_profile is not None:
            try:
                profile = load_agent_profile(self._agent_profile)
            except Exception as e:
                raise ProviderError(f"Failed to load agent profile '{self._agent_profile}': {e}")

        command_parts = ["feynman"]

        # Per-call override wins over static profile model
        resolved_model = self._model or (profile.model if profile else None)
        if resolved_model:
            command_parts.extend(["--model", resolved_model])

        if profile and getattr(profile, "feynmanConfig", None):
            config = profile.feynmanConfig
            if isinstance(config, dict):
                thinking = config.get("thinking")
                if thinking:
                    command_parts.extend(["--thinking", str(thinking)])

        if self._skill_prompt:
            logger.warning(
                "Feynman provider runs in its own research runtime; "
                "configure skills and MCP servers inside Feynman's workbench or skills folder"
            )

        if self._allowed_tools and "*" not in self._allowed_tools:
            logger.warning(
                "Feynman provider has no CAO-native tool restriction flag; "
                "tool usage is governed by Feynman's internal subagents"
            )

        return shlex.join(command_parts)

    async def initialize(self) -> bool:
        """Initialize Feynman by starting the CLI chat REPL."""
        init_timeout = get_server_settings()["provider_init_timeout"]
        if not await wait_for_shell(self.terminal_id, timeout=init_timeout):
            raise TimeoutError(f"Shell initialization timed out after {init_timeout}s")

        command = self._build_feynman_command()
        get_backend().send_keys(self.session_name, self.window_name, command)

        # Early check for setup wizard or immediate failures before HTTP timeout
        import asyncio
        import time
        from cli_agent_orchestrator.services.status_monitor import status_monitor

        start = time.time()
        timeout = min(60.0, float(init_timeout))
        while time.time() - start < timeout:
            current = await asyncio.to_thread(status_monitor.get_status, self.terminal_id)
            if current in {TerminalStatus.IDLE, TerminalStatus.COMPLETED}:
                self._initialized = True
                return True
            pane_output = get_backend().get_history(self.session_name, self.window_name)
            if re.search(SETUP_REQUIRED_PATTERN, pane_output):
                raise ProviderError(
                    "Feynman requires model access configuration before first launch. "
                    "Please run 'feynman setup' in your terminal or configure an API key "
                    "(e.g. GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY)."
                )
            if current == TerminalStatus.ERROR:
                raise ProviderError(f"Feynman initialization failed: {pane_output[-300:].strip()}")
            await asyncio.sleep(0.5)

        pane_output = get_backend().get_history(self.session_name, self.window_name)
        if re.search(SETUP_REQUIRED_PATTERN, pane_output):
            raise ProviderError(
                "Feynman requires model access configuration before first launch. "
                "Please run 'feynman setup' in your terminal or configure an API key "
                "(e.g. GEMINI_API_KEY, ANTHROPIC_API_KEY, OPENAI_API_KEY)."
            )

        raise TimeoutError(f"Feynman initialization timed out after {timeout}s")

    def get_status(self, output: str) -> TerminalStatus:
        """Get Feynman status by analyzing the terminal output buffer."""
        native = self._resolve_native_status(output)
        if native is not None:
            return native

        output = self._resolve_buffer(output)
        if not output:
            return TerminalStatus.ERROR

        clean_output = _strip_ansi(output)
        if re.search(SETUP_REQUIRED_PATTERN, clean_output):
            return TerminalStatus.ERROR

        lines = clean_output.splitlines()
        tail_lines_text = lines[-30:]
        tail_output = "\n".join(tail_lines_text)
        bottom_lines_text = [line for line in tail_lines_text if line.strip()][-8:]
        bottom_output = "\n".join(bottom_lines_text)
        has_idle_prompt = any(_is_idle_line(line.strip()) for line in bottom_lines_text)
        has_stable_idle_timer = self._has_stable_idle_timer(tail_output)
        has_turn_completed = (
            bool(re.search(r"↑\d+k?\s*↓\d+k?", clean_output))
            or bool(re.search(USER_PREFIX_PATTERN, clean_output, re.MULTILINE))
        )
        has_response = bool(re.search(ASSISTANT_HEADER_PATTERN, clean_output))
        if not has_response:
            has_response = self._has_extractable_response(clean_output)

        if _has_waiting_prompt(bottom_output):
            return TerminalStatus.WAITING_USER_ANSWER

        if re.search(ERROR_PATTERN, tail_output, re.IGNORECASE | re.MULTILINE):
            return TerminalStatus.ERROR

        if re.search(ACTIVE_PROCESSING_PATTERN, bottom_output, re.IGNORECASE | re.MULTILINE):
            return TerminalStatus.PROCESSING

        if has_stable_idle_timer or has_idle_prompt:
            if has_turn_completed and has_response:
                return TerminalStatus.COMPLETED
            return TerminalStatus.IDLE

        if re.search(PROCESSING_PATTERN, bottom_output, re.IGNORECASE | re.MULTILINE):
            return TerminalStatus.PROCESSING

        return TerminalStatus.PROCESSING

    def get_idle_pattern_for_log(self) -> str:
        """Return Feynman idle prompt pattern for log file monitoring."""
        return IDLE_PROMPT_PATTERN_LOG

    def _has_stable_idle_timer(self, tail_output: str) -> bool:
        """Detect settled Feynman output using the status bar idle timer when available."""
        current_timer = _last_idle_timer(tail_output)
        if current_timer is None:
            self._last_idle_timer = None
            self._stable_idle_timer_count = 0
            return False

        if current_timer == self._last_idle_timer:
            self._stable_idle_timer_count += 1
        else:
            self._last_idle_timer = current_timer
            self._stable_idle_timer_count = 1

        return 2 <= self._stable_idle_timer_count <= MAX_STABLE_IDLE_TIMER_POLLS

    def _has_extractable_response(self, clean_output: str) -> bool:
        try:
            return bool(self._extract_response(clean_output, require_header=False))
        except ValueError:
            return False

    def _extract_response(self, clean_output: str, require_header: bool = True) -> str:
        matches = list(re.finditer(ASSISTANT_HEADER_PATTERN, clean_output))
        if matches:
            start = clean_output.find("\n", matches[-1].end())
            if start == -1:
                raise ValueError("No Feynman response content found")
            start += 1
            search_region = clean_output[start:]
        elif require_header:
            raise ValueError("No Feynman response found - no assistant header detected")
        else:
            user_matches = list(re.finditer(USER_PREFIX_PATTERN, clean_output, re.MULTILINE))
            if user_matches:
                user_line_end = clean_output.find("\n", user_matches[-1].end())
                if user_line_end == -1:
                    user_line_end = user_matches[-1].end()
                search_region = clean_output[user_line_end + 1 :]
            else:
                search_region = clean_output

        end_match = re.search(IDLE_PROMPT_PATTERN, search_region, re.MULTILINE)
        candidate_text = search_region[: end_match.start()] if end_match else search_region

        paragraphs = [p.strip() for p in candidate_text.split("\n\n") if p.strip()]
        for p in reversed(paragraphs):
            lines = p.splitlines()
            if any("v0.3." in l or "╭──" in l or "╰──" in l for l in lines):
                continue
            if re.search(r"^(?:Translating|Thinking|musing|Searching|Fetching|Synthesizing)", p):
                continue
            if _is_chrome_line(p):
                continue
            return p

        candidate_lines = candidate_text.splitlines()
        response_lines: list[str] = []
        for raw_line in reversed(candidate_lines):
            line = raw_line.rstrip()
            stripped = line.strip()
            if _is_chrome_line(stripped):
                if response_lines and stripped:
                    break
                continue
            response_lines.append(stripped)

        response = "\n".join(reversed(response_lines)).strip()
        if not response:
            raise ValueError("Empty Feynman response - no content found")
        return response

    def extract_last_message_from_script(self, script_output: str) -> str:
        """Extract the last Feynman response from terminal output."""
        clean_output = _strip_ansi(script_output)
        return self._extract_response(clean_output, require_header=False)

    def exit_cli(self) -> str:
        """Get the command to exit Feynman."""
        return "/exit"

    def cleanup(self) -> None:
        """Clean up Feynman provider state."""
        self._initialized = False
