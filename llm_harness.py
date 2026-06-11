import os
import time
import uuid
import json
import tiktoken
from datetime import datetime
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv
from groq import Groq
import groq as groq_errors

load_dotenv()

enc = tiktoken.get_encoding("cl100k_base")

# ── Response dataclass ─────────────────────────────────────────────────────
@dataclass
class HarnessResponse:
    """
    Structured response object returned by every harness call.
    Callers always receive this — never a raw API response object.
    """
    text:          str              # the response text
    parsed:        Optional[dict]   # parsed JSON if call_json was used
    model:         str              # which model actually ran
    input_tokens:  int
    output_tokens: int
    latency_ms:    int
    cost_usd:      float
    call_id:       str
    finish_reason: str
    was_fallback:  bool             # True if fallback model was used
    call_type:     str

# ── The Harness class ──────────────────────────────────────────────────────
class LLMHarness:
    """
    Central wrapper for all LLM API calls.
    Manages: input validation, routing, retries, fallback,
             structured logging, output parsing.

    Usage:
        harness = LLMHarness()
        result  = harness.call(messages, task_type="analysis")
        print(result.text)
    """

    # Pricing per million tokens (input, output)
    PRICING = {
        "llama-3.3-70b-versatile": (0.59, 0.79),
        "llama-3.1-8b-instant":    (0.05, 0.08),
    }

    def __init__(
        self,
        primary_model:    str = "llama-3.3-70b-versatile",
        fallback_model:   str = "llama-3.1-8b-instant",
        max_input_tokens: int = 7000,
        max_retries:      int = 3,
        log_to_console:   bool = True,
    ):
        self.client           = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.primary_model    = primary_model
        self.fallback_model   = fallback_model
        self.cheap_model      = fallback_model
        self.max_input_tokens = max_input_tokens
        self.max_retries      = max_retries
        self.log_to_console   = log_to_console
        self._call_log        = []
        # In-memory call log — in production send to logging service

    # ── Token counting ─────────────────────────────────────────────────────
    def count_tokens(self, messages: list[dict]) -> int:
        """Count total tokens across all messages including formatting overhead."""
        return sum(
            len(enc.encode(m["content"])) + 4
            for m in messages
        )

    # ── Model routing ──────────────────────────────────────────────────────
    def route_model(self, task_type: str, input_tokens: int) -> str:
        """Select model based on task complexity and input size."""
        if input_tokens > 6000:
            # Large input — use powerful model
            return self.primary_model

        simple_tasks = [
            "classification", "sentiment", "extraction",
            "formatting", "routing", "labeling"
        ]
        if task_type in simple_tasks:
            return self.cheap_model

        return self.primary_model

    # ── Input validation ───────────────────────────────────────────────────
    def validate_input(self, messages: list[dict]) -> None:
        """Validate messages before sending to API. Raises ValueError on failure."""
        if not messages:
            raise ValueError("Messages list cannot be empty.")

        if messages[-1]["role"] != "user":
            raise ValueError(
                f"Last message must have role 'user', got '{messages[-1]['role']}'"
            )

        total_tokens = self.count_tokens(messages)
        if total_tokens > self.max_input_tokens:
            raise ValueError(
                f"Input token count {total_tokens} exceeds limit {self.max_input_tokens}. "
                f"Apply context compression before calling."
            )

        for i, msg in enumerate(messages):
            if "role" not in msg or "content" not in msg:
                raise ValueError(f"Message {i} missing 'role' or 'content' key.")
            if not isinstance(msg["content"], str):
                raise ValueError(f"Message {i} content must be a string.")

    # ── Cost calculation ───────────────────────────────────────────────────
    def calculate_cost(
        self, model: str, input_tokens: int, output_tokens: int
    ) -> float:
        """Calculate cost in USD for this call."""
        input_price, output_price = self.PRICING.get(model, (0.0, 0.0))
        return round(
            (input_tokens  / 1_000_000 * input_price) +
            (output_tokens / 1_000_000 * output_price),
            6
        )

    # ── Structured logging ─────────────────────────────────────────────────
    def log_call(self, entry: dict) -> None:
        """Log a call entry. Stores in memory and optionally prints."""
        self._call_log.append(entry)
        if self.log_to_console:
            print(f"[HARNESS LOG] {json.dumps(entry)}")

    def get_call_log(self) -> list[dict]:
        """Return all logged calls for this harness instance."""
        return self._call_log

    # ── Core API call (single attempt) ────────────────────────────────────
    def _single_call(
        self,
        messages:         list[dict],
        model:            str,
        max_tokens:       int,
        response_format:  Optional[dict] = None,
    ) -> tuple:
        """
        Makes one API call. Returns (response_text, usage, finish_reason).
        Does not handle retries — that is the caller's responsibility.
        """
        kwargs = {
            "model":      model,
            "max_tokens": max_tokens,
            "messages":   messages,
        }
        if response_format:
            kwargs["response_format"] = response_format

        response = self.client.chat.completions.create(**kwargs)

        text         = response.choices[0].message.content
        finish_reason = response.choices[0].finish_reason
        usage        = response.usage

        return text, usage, finish_reason

    # ── Main call method ───────────────────────────────────────────────────
    def call(
        self,
        messages:   list[dict],
        task_type:  str = "general",
        max_tokens: int = 512,
    ) -> HarnessResponse:
        """
        The primary interface for all LLM calls.
        Handles validation, routing, retries, fallback, and logging.
        """
        # Step 1 — Validate input
        self.validate_input(messages)

        # Step 2 — Route to appropriate model
        input_tokens = self.count_tokens(messages)
        model        = self.route_model(task_type, input_tokens)
        call_id      = str(uuid.uuid4())[:8]
        start_time   = time.time()
        was_fallback = False
        error_msg    = None

        # Step 3 — Try primary model with retries
        text = finish_reason = usage = None

        for attempt in range(self.max_retries):
            try:
                text, usage, finish_reason = self._single_call(
                    messages, model, max_tokens
                )
                break   # success — exit retry loop

            except groq_errors.AuthenticationError as e:
                # Permanent error — do not retry
                raise RuntimeError(f"Authentication failed: {e}") from e

            except groq_errors.BadRequestError as e:
                # Permanent error — do not retry
                raise RuntimeError(f"Bad request: {e}") from e

            except (
                groq_errors.RateLimitError,
                groq_errors.APITimeoutError,
                groq_errors.APIConnectionError,
            ) as e:
                if attempt == self.max_retries - 1:
                    # All retries exhausted — try fallback
                    print(f"[HARNESS] Primary model failed after {self.max_retries} attempts. Trying fallback.")
                    try:
                        model        = self.fallback_model
                        was_fallback = True
                        text, usage, finish_reason = self._single_call(
                            messages, model, max_tokens
                        )
                    except Exception as fallback_error:
                        error_msg = str(fallback_error)
                        raise RuntimeError(
                            f"Both primary and fallback failed. Last error: {fallback_error}"
                        ) from fallback_error
                else:
                    wait = 2 ** attempt
                    print(f"[HARNESS] Attempt {attempt+1} failed. Retrying in {wait}s...")
                    time.sleep(wait)

        # Step 4 — Check finish reason
        if finish_reason == "length":
            print(f"[HARNESS WARNING] Response truncated — increase max_tokens (currently {max_tokens})")

        # Step 5 — Build response object
        latency_ms    = round((time.time() - start_time) * 1000)
        output_tokens = usage.completion_tokens if usage else 0
        input_tokens_actual = usage.prompt_tokens if usage else input_tokens
        cost          = self.calculate_cost(model, input_tokens_actual, output_tokens)

        response = HarnessResponse(
            text=text,
            parsed=None,
            model=model,
            input_tokens=input_tokens_actual,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            call_id=call_id,
            finish_reason=finish_reason,
            was_fallback=was_fallback,
            call_type=task_type,
        )

        # Step 6 — Log the call
        self.log_call({
            "call_id":       call_id,
            "timestamp":     datetime.utcnow().isoformat(),
            "task_type":     task_type,
            "model":         model,
            "was_fallback":  was_fallback,
            "input_tokens":  input_tokens_actual,
            "output_tokens": output_tokens,
            "latency_ms":    latency_ms,
            "cost_usd":      cost,
            "finish_reason": finish_reason,
            "error":         error_msg,
        })

        return response

    # ── JSON call method ───────────────────────────────────────────────────
    def call_json(
        self,
        messages:   list[dict],
        task_type:  str = "extraction",
        max_tokens: int = 512,
    ) -> HarnessResponse:
        """
        Like call() but enforces JSON output and parses the response.
        Returns HarnessResponse with .parsed populated as a Python dict.
        """
        # Validate input
        self.validate_input(messages)

        input_tokens = self.count_tokens(messages)
        model        = self.route_model(task_type, input_tokens)
        call_id      = str(uuid.uuid4())[:8]
        start_time   = time.time()
        error_msg    = None

        text, usage, finish_reason = self._single_call(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

        # Parse JSON response
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as e:
            # Clean markdown fences if present
            cleaned = text.strip()
            if cleaned.startswith("```json"):
                cleaned = cleaned[7:]
            if cleaned.startswith("```"):
                cleaned = cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[:-3]
            try:
                parsed = json.loads(cleaned.strip())
            except json.JSONDecodeError:
                parsed = {"error": "json_parse_failed", "raw": text}

        latency_ms    = round((time.time() - start_time) * 1000)
        output_tokens = usage.completion_tokens
        cost          = self.calculate_cost(model, usage.prompt_tokens, output_tokens)

        response = HarnessResponse(
            text=text,
            parsed=parsed,
            model=model,
            input_tokens=usage.prompt_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            cost_usd=cost,
            call_id=call_id,
            finish_reason=finish_reason,
            was_fallback=False,
            call_type=task_type,
        )

        self.log_call({
            "call_id":       call_id,
            "timestamp":     datetime.utcnow().isoformat(),
            "task_type":     task_type,
            "model":         model,
            "input_tokens":  usage.prompt_tokens,
            "output_tokens": output_tokens,
            "latency_ms":    latency_ms,
            "cost_usd":      cost,
            "finish_reason": finish_reason,
            "error":         error_msg,
        })

        return response

    # ── Session cost summary ───────────────────────────────────────────────
    def session_summary(self) -> dict:
        """Return total cost and token usage across all calls this session."""
        total_cost    = sum(e["cost_usd"]      for e in self._call_log)
        total_input   = sum(e["input_tokens"]  for e in self._call_log)
        total_output  = sum(e["output_tokens"] for e in self._call_log)
        total_calls   = len(self._call_log)
        fallback_calls = sum(1 for e in self._call_log if e.get("was_fallback"))

        return {
            "total_calls":    total_calls,
            "fallback_calls": fallback_calls,
            "total_input_tokens":  total_input,
            "total_output_tokens": total_output,
            "total_cost_usd": round(total_cost, 6),
            "avg_latency_ms": round(
                sum(e["latency_ms"] for e in self._call_log) / max(total_calls, 1)
            ),
        }