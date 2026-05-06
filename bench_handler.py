from __future__ import annotations

import json
import os
import re
import sys
import time
import httpx
from copy import deepcopy
from pathlib import Path

BENCH_ROOT = Path(__file__).parent.resolve()
BFCL_ROOT = Path("/root/gorilla/berkeley-function-call-leaderboard")
BFCL_COMP_ROOT = Path("/root/bfcl_compression_bench")
sys.path.insert(0, str(BFCL_ROOT))
sys.path.insert(0, str(BFCL_COMP_ROOT))
sys.path.insert(0, str(BENCH_ROOT))

from bfcl_eval.constants.default_prompts import (  # noqa: E402
    DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
    MAXIMUM_STEP_LIMIT,
)
from bfcl_eval.constants.enums import ModelStyle  # noqa: E402
from bfcl_eval.constants.type_mappings import GORILLA_TO_OPENAPI  # noqa: E402
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (  # noqa: E402
    execute_multi_turn_func_call,
    is_empty_execute_response,
)
from bfcl_eval.model_handler.utils import convert_to_tool  # noqa: E402
from compressor import ContextCompressor, should_compact  # noqa: E402
from llama_handler import (  # noqa: E402
    _coerce_tool_call_args,
    _convert_to_function_call_coerced,
    split_multi_tool_calls,
)
from openai import BadRequestError, OpenAI  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

import sp_config as CFG  # noqa: E402


def run_dirs(run_name: str) -> dict:
    base = CFG.RESULTS_DIR / run_name
    return {
        "base": base,
        "traces": base / "traces",
        "abc": base / "abc_segments",
        "timing": base / "timing",
        "model_results": base / "model_results",
        "prompt_logs": base / "prompt_logs",
        "checkpoints": base / "checkpoints",
        "config_file": base / "run_config.json",
        "samples_file": base / "sample_ids.json",
        "analysis_file": base / "analysis.json",
        "summary_file": base / "summary.jsonl",
    }


def ensure_run_dirs(dirs: dict) -> None:
    for key, path in dirs.items():
        if key.endswith("_file"):
            continue
        path.mkdir(parents=True, exist_ok=True)


def resolve_preset(name: str) -> dict:
    for preset_name, context_window, reserve_tokens, keep_recent_tokens, summary_max_tokens in CFG.PRESETS:
        if preset_name == name:
            return {
                "name": preset_name,
                "context_window": context_window,
                "reserve_tokens": reserve_tokens,
                "keep_recent_tokens": keep_recent_tokens,
                "summary_max_tokens": summary_max_tokens,
                "threshold": context_window - reserve_tokens,
            }
    raise KeyError(f"unknown preset: {name}")


def _split_before(messages: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    if not messages:
        return [], [], []
    if messages[0].get("role") == "system":
        return [messages[0]], list(messages[1:]), []
    return [], list(messages), []


def _split_after(messages: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    if not messages:
        return [], [], []
    if messages[0].get("role") == "system":
        a = [messages[0]]
        rest = list(messages[1:])
    else:
        a = []
        rest = list(messages)

    b2 = []
    index = 0
    while index < len(rest):
        msg = rest[index]
        content = str(msg.get("content", ""))
        if msg.get("role") == "user" and "[Previous conversation summary]" in content:
            b2.append(msg)
            index += 1
            continue
        if msg.get("role") == "assistant" and "context from the previous conversation" in content:
            b2.append(msg)
            index += 1
            continue
        break

    return a, b2, rest[index:]


def refine_before_c1(abc_event: dict) -> None:
    before = abc_event["abc_segments"]["before"]
    after = abc_event["abc_segments"]["after"]
    keep_len = len(after["C1"])
    if keep_len <= 0 or len(before["B1"]) < keep_len:
        return
    before["C1"] = before["B1"][-keep_len:]
    before["B1"] = before["B1"][:-keep_len]


def _is_retryable_stream_error(exc: Exception) -> bool:
    return isinstance(
        exc,
        (
            httpx.RemoteProtocolError,
            httpx.ReadError,
            httpx.ReadTimeout,
            httpx.WriteError,
            httpx.ConnectError,
        ),
    )


MARKET_PRICE_PATTERN = re.compile(
    r"\b(?:current|prevailing|present|going)\s+market\s+(?:price|rate|value)\b|\bmarket\s+(?:price|rate|value)\b",
    re.IGNORECASE,
)


def _looks_like_market_price(value: object) -> bool:
    if not isinstance(value, str):
        return False
    normalized = " ".join(value.replace("_", " ").replace("-", " ").split())
    return bool(MARKET_PRICE_PATTERN.search(normalized))


def _extract_price_from_stock_info(result: str) -> float | None:
    try:
        payload = json.loads(result)
    except (TypeError, json.JSONDecodeError):
        return None
    price = payload.get("price")
    try:
        return float(price)
    except (TypeError, ValueError):
        return None


class BFCLLongContextSemiPrefillBench:
    def __init__(
        self,
        run_name: str,
        preset: dict,
        model_key: str = CFG.DEFAULT_MODEL,
        compression_enabled: bool = True,
    ):
        self.run_name = run_name
        self.preset = preset
        self.compression_enabled = compression_enabled
        self.model_key = model_key
        self.model_info = CFG.MODEL_REGISTRY[model_key]
        self.model_path = self.model_info["model_path"]
        self.tokenizer_path = self.model_info["tokenizer_path"]
        self.dirs = run_dirs(run_name)
        ensure_run_dirs(self.dirs)

        self.client = OpenAI(api_key=CFG.API_KEY, base_url=CFG.PROXY_URL)
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path, trust_remote_code=True)
        self.compressor = ContextCompressor(
            api_base=CFG.PROXY_URL,
            model_name=self.model_path,
            api_key=CFG.API_KEY,
            summary_max_tokens=preset["summary_max_tokens"],
            context_window=preset["context_window"],
            reserve_tokens=preset["reserve_tokens"],
            keep_recent_tokens=preset["keep_recent_tokens"],
            quality_guard_enabled=False,
            quality_guard_max_retries=CFG.SUMMARY_MAX_RETRIES,
            use_structured_instructions=CFG.USE_STRUCTURED_INSTRUCTIONS,
            preserved_recent_turns=CFG.PRESERVED_RECENT_TURNS,
        )
        self.model_name_underline_replaced = self.model_path.replace("/", "_").replace("-", "_").replace(".", "_")

    def load_checkpoint(self, sample_id: str) -> dict | None:
        path = self.dirs["checkpoints"] / f"{sample_id}.json"
        if not path.exists():
            return None
        with open(path, "r", encoding="utf-8") as handle:
            return json.load(handle)

    def clear_checkpoint(self, sample_id: str) -> None:
        path = self.dirs["checkpoints"] / f"{sample_id}.json"
        if path.exists():
            path.unlink()

    def sample_finished(self, sample_id: str) -> bool:
        trace_path = self.dirs["traces"] / f"{sample_id}.json"
        checkpoint_path = self.dirs["checkpoints"] / f"{sample_id}.json"
        return trace_path.exists() and not checkpoint_path.exists()

    def _normalize_messages(self, messages: list[dict]) -> list[dict]:
        normalized = []
        for msg in messages:
            if hasattr(msg, "model_dump"):
                item = msg.model_dump(exclude_none=True)
            else:
                item = deepcopy(msg)
            if item.get("tool_calls"):
                tool_calls = []
                for tool_call in item["tool_calls"]:
                    if hasattr(tool_call, "model_dump"):
                        tool_calls.append(tool_call.model_dump(exclude_none=True))
                    else:
                        tool_calls.append(deepcopy(tool_call))
                item["tool_calls"] = tool_calls
            normalized.append(item)
        return normalized

    def _message_text(self, msg: dict) -> str:
        parts = [str(msg.get("role", ""))]
        if msg.get("content"):
            parts.append(str(msg["content"]))
        if msg.get("tool_calls"):
            parts.append(json.dumps(msg["tool_calls"], ensure_ascii=False))
        if msg.get("tool_call_id"):
            parts.append(str(msg["tool_call_id"]))
        return "\n".join(parts)

    def message_tokens(self, messages: list[dict]) -> int:
        if not messages:
            return 0
        text = "\n".join(self._message_text(msg) for msg in self._normalize_messages(messages))
        return len(self.tokenizer.tokenize(text))

    def tools_tokens(self, tools: list[dict]) -> int:
        if not tools:
            return 0
        return len(self.tokenizer.tokenize(json.dumps(tools, ensure_ascii=False)))

    def prompt_tokens(self, messages: list[dict], tools: list[dict]) -> int:
        return self.message_tokens(messages) + self.tools_tokens(tools)

    def _extract_previous_summary(self, messages: list[dict]) -> str | None:
        for msg in messages:
            content = str(msg.get("content", ""))
            marker = "[Previous conversation summary]\n"
            if msg.get("role") != "user" or marker not in content:
                continue
            start = content.find(marker) + len(marker)
            end = content.find("\n[End of summary.")
            if end > start:
                return content[start:end]
        return None

    def _msg_list_tokens(self, messages: list[dict]) -> int:
        return self.message_tokens(messages)

    def _msg_token_list(self, messages: list[dict]) -> list[int]:
        return [self.message_tokens([message]) for message in messages]

    def _build_tools(self, test_entry: dict) -> list[dict]:
        return convert_to_tool(test_entry["function"], GORILLA_TO_OPENAPI, ModelStyle.OPENAI_COMPLETIONS)

    def _classify_step(self, state: dict) -> str:
        if state["turn_idx"] == 0 and state["step_idx"] == 0:
            return "full_prefill"
        if state.get("last_was_compression"):
            return "semi_prefill"
        return "incremental"

    def _append_prompt_log(self, sample_id: str, entry: dict) -> None:
        path = self.dirs["prompt_logs"] / f"{sample_id}.jsonl"
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def _persist_sample_logs(self, state: dict) -> None:
        sample_id = state["sample_id"]
        with open(self.dirs["timing"] / f"{sample_id}.json", "w", encoding="utf-8") as handle:
            json.dump(state["timing_log"], handle, indent=2, ensure_ascii=False)
        with open(self.dirs["abc"] / f"{sample_id}.json", "w", encoding="utf-8") as handle:
            json.dump(state["abc_snapshots"], handle, indent=2, ensure_ascii=False)
        with open(self.dirs["traces"] / f"{sample_id}.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "test_id": sample_id,
                    "turns": state["turn_traces"],
                    "total_turns": state["total_turns"],
                    "steps": len(state["timing_log"]),
                    "compressions": len(state["abc_snapshots"]),
                    "tool_calls": state["cumulative_tool_calls"],
                    "prompt_snapshots": state["prompt_snapshots"],
                    "resume_count": state.get("resume_count", 0),
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )
        with open(self.dirs["model_results"] / f"{sample_id}.json", "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "id": sample_id,
                    "result": state.get("official_model_result", []),
                },
                handle,
                indent=2,
                ensure_ascii=False,
            )

    def _save_checkpoint(self, state: dict, test_entry: dict) -> None:
        payload = {
            "sample_id": state["sample_id"],
            "turn_idx": state["turn_idx"],
            "step_idx": state["step_idx"],
            "messages": self._normalize_messages(state["messages"]),
            "tools": state["tools"],
            "timing_log": state["timing_log"],
            "abc_snapshots": state["abc_snapshots"],
            "prompt_snapshots": state["prompt_snapshots"],
            "turn_traces": state["turn_traces"],
            "official_model_result": state.get("official_model_result", []),
            "executed_calls_history": state["executed_calls_history"],
            "cumulative_tool_calls": state["cumulative_tool_calls"],
            "previous_summary": state.get("previous_summary"),
            "last_was_compression": state.get("last_was_compression", False),
            "total_turns": state["total_turns"],
            "resume_count": state.get("resume_count", 0),
            "saved_at": time.time(),
            "test_entry_id": test_entry["id"],
        }
        target = self.dirs["checkpoints"] / f"{state['sample_id']}.json"
        tmp = target.with_suffix(".json.tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False)
        os.replace(tmp, target)

    def _rebuild_tool_state(self, test_entry: dict, state: dict) -> None:
        initial_config = test_entry.get("initial_config", {})
        involved_classes = test_entry["involved_classes"]
        sample_id = test_entry["id"]
        long_context = "long_context" in sample_id or "composite" in sample_id
        execute_multi_turn_func_call(
            [],
            initial_config,
            involved_classes,
            self.model_name_underline_replaced,
            sample_id,
            long_context=long_context,
            is_evaL_run=False,
        )
        for calls in state.get("executed_calls_history", []):
            if not calls:
                continue
            execute_multi_turn_func_call(
                calls,
                initial_config,
                involved_classes,
                self.model_name_underline_replaced,
                sample_id,
                long_context=long_context,
                is_evaL_run=False,
            )

    def _maybe_compress(self, state: dict) -> None:
        if not self.compression_enabled:
            return
        total_tokens = self.prompt_tokens(state["messages"], state["tools"])
        if not should_compact(total_tokens, self.preset["context_window"], self.preset["reserve_tokens"]):
            return

        before_messages = self._normalize_messages(state["messages"])
        before_prompt_tokens = self.message_tokens(before_messages)
        compressed_messages, info = self.compressor.compress(
            before_messages,
            keep_recent_turns=1,
            previous_summary=state.get("previous_summary"),
            use_token_budget=True,
        )
        if info is None:
            return

        state["messages"] = compressed_messages
        state["previous_summary"] = self._extract_previous_summary(compressed_messages)

        a_before, b1_before, c1_before = _split_before(before_messages)
        a_after, b2_after, c1_after = _split_after(compressed_messages)

        abc_event = {
            "turn": state["turn_idx"],
            "step": state["step_idx"],
            "pre_prompt_tokens": before_prompt_tokens,
            "post_prompt_tokens": self.message_tokens(compressed_messages),
            "summary_generation_time_s": info.get("summary_generation_time_s"),
            "abc_segments": {
                "before": {
                    "A": a_before,
                    "B1": b1_before,
                    "C1": c1_before,
                },
                "after": {
                    "A": a_after,
                    "B2": b2_after,
                    "C1": c1_after,
                },
            },
        }
        refine_before_c1(abc_event)

        before_section = abc_event["abc_segments"]["before"]
        after_section = abc_event["abc_segments"]["after"]
        abc_event["A_tokens"] = self._msg_list_tokens(before_section["A"])
        abc_event["B1_tokens"] = self._msg_list_tokens(before_section["B1"])
        abc_event["B2_tokens"] = self._msg_list_tokens(after_section["B2"])
        abc_event["C1_tokens_before"] = self._msg_list_tokens(before_section["C1"])
        abc_event["C1_tokens_after"] = self._msg_list_tokens(after_section["C1"])
        abc_event["B2_plus_C1_tokens"] = abc_event["B2_tokens"] + abc_event["C1_tokens_after"]
        abc_event["c1_below_minimum"] = abc_event["C1_tokens_after"] < CFG.C1_MIN_TOKENS
        abc_event["c1_at_or_below_minimum"] = abc_event["C1_tokens_after"] <= CFG.C1_MIN_TOKENS
        abc_event["c1_above_diagnostic_maximum"] = abc_event["C1_tokens_after"] > CFG.C1_MAX_TOKENS
        abc_event["c1_meets_minimum"] = abc_event["C1_tokens_after"] > CFG.C1_MIN_TOKENS
        abc_event["c1_in_target_range"] = abc_event["c1_meets_minimum"]

        before_section["A_tokens"] = abc_event["A_tokens"]
        before_section["B1_tokens"] = abc_event["B1_tokens"]
        before_section["C1_tokens"] = abc_event["C1_tokens_before"]
        before_section["A_message_tokens"] = self._msg_token_list(before_section["A"])
        before_section["B1_message_tokens"] = self._msg_token_list(before_section["B1"])
        before_section["C1_message_tokens"] = self._msg_token_list(before_section["C1"])
        after_section["A_tokens"] = abc_event["A_tokens"]
        after_section["B2_tokens"] = abc_event["B2_tokens"]
        after_section["C1_tokens"] = abc_event["C1_tokens_after"]
        after_section["A_message_tokens"] = self._msg_token_list(after_section["A"])
        after_section["B2_message_tokens"] = self._msg_token_list(after_section["B2"])
        after_section["C1_message_tokens"] = self._msg_token_list(after_section["C1"])

        state["abc_snapshots"].append(abc_event)
        state["prompt_snapshots"].append(
            {
                "type": "compression",
                "turn": state["turn_idx"],
                "step": state["step_idx"],
                "pre_prompt_tokens": abc_event["pre_prompt_tokens"],
                "post_prompt_tokens": abc_event["post_prompt_tokens"],
                "B2_plus_C1_tokens": abc_event["B2_plus_C1_tokens"],
                "C1_tokens_after": abc_event["C1_tokens_after"],
                "summary_generation_time_s": abc_event["summary_generation_time_s"],
            }
        )
        self._append_prompt_log(
            state["sample_id"],
            {
                "type": "compression",
                "turn": state["turn_idx"],
                "step": state["step_idx"],
                "messages_before": before_messages,
                "messages_after": compressed_messages,
                "abc": abc_event["abc_segments"],
                "pre_prompt_tokens": abc_event["pre_prompt_tokens"],
                "post_prompt_tokens": abc_event["post_prompt_tokens"],
            },
        )
        state["last_was_compression"] = True

    def _maybe_compress_before_query(self, state: dict) -> None:
        # Turn-boundary compression is handled in run_sample; this covers long tool loops within a turn.
        if state["step_idx"] <= 0:
            return
        self._maybe_compress(state)

    def _stream_query(self, state: dict) -> tuple[dict, dict]:
        sample_id = state["sample_id"]
        self._maybe_compress_before_query(state)
        messages = split_multi_tool_calls(self._normalize_messages(state["messages"]))
        tools = state["tools"]
        classification = self._classify_step(state)
        prompt_tokens = self.prompt_tokens(messages, tools)
        tool_schema_tokens = self.tools_tokens(tools)

        self._append_prompt_log(
            sample_id,
            {
                "type": "query",
                "turn": state["turn_idx"],
                "step": state["step_idx"],
                "classification": classification,
                "prompt_tokens": prompt_tokens,
                "tool_schema_tokens": tool_schema_tokens,
                "messages": messages,
                "tools": tools,
            },
        )
        state["prompt_snapshots"].append(
            {
                "type": "query",
                "turn": state["turn_idx"],
                "step": state["step_idx"],
                "classification": classification,
                "prompt_tokens": prompt_tokens,
                "tool_schema_tokens": tool_schema_tokens,
                "message_count": len(messages),
            }
        )

        request = {
            "model": self.model_path,
            "messages": messages,
            "temperature": CFG.TEMPERATURE,
            "store": False,
            "stream": True,
        }
        if tools:
            request["tools"] = tools

        max_attempts = max(1, int(getattr(CFG, "STREAM_MAX_RETRIES", 0)) + 1)
        backoff_s = float(getattr(CFG, "STREAM_RETRY_BACKOFF_S", 0.0))

        for attempt_idx in range(max_attempts):
            t0 = time.perf_counter()
            ttft_ms = None
            content_parts = []
            tool_calls_by_index = {}
            usage = None
            stream = None

            try:
                try:
                    stream = self.client.chat.completions.create(
                        **request,
                        stream_options={"include_usage": True},
                    )
                except TypeError:
                    stream = self.client.chat.completions.create(**request)

                for chunk in stream:
                    if getattr(chunk, "usage", None):
                        usage = chunk.usage
                    if not getattr(chunk, "choices", None):
                        continue
                    delta = getattr(chunk.choices[0], "delta", None)
                    if delta is None:
                        continue
                    if getattr(delta, "content", None):
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t0) * 1000.0
                        content_parts.append(delta.content)
                    for tool_call in getattr(delta, "tool_calls", []) or []:
                        if ttft_ms is None:
                            ttft_ms = (time.perf_counter() - t0) * 1000.0
                        index = getattr(tool_call, "index", 0) or 0
                        entry = tool_calls_by_index.setdefault(
                            index,
                            {
                                "id": "",
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        if getattr(tool_call, "id", None):
                            entry["id"] = tool_call.id
                        if getattr(tool_call, "type", None):
                            entry["type"] = tool_call.type
                        function = getattr(tool_call, "function", None)
                        if function is not None:
                            if getattr(function, "name", None):
                                entry["function"]["name"] += function.name
                            if getattr(function, "arguments", None):
                                entry["function"]["arguments"] += function.arguments

                total_ms = (time.perf_counter() - t0) * 1000.0
                if ttft_ms is None:
                    ttft_ms = total_ms

                ordered_tool_calls = [tool_calls_by_index[idx] for idx in sorted(tool_calls_by_index)]
                content = "".join(content_parts)
                output_token_fallback_text = content + json.dumps(ordered_tool_calls, ensure_ascii=False)
                output_tokens = len(self.tokenizer.tokenize(output_token_fallback_text)) if output_token_fallback_text else 0
                input_tokens = prompt_tokens
                if usage is not None:
                    input_tokens = getattr(usage, "prompt_tokens", prompt_tokens) or prompt_tokens
                    output_tokens = getattr(usage, "completion_tokens", output_tokens) or output_tokens

                assistant_message = {"role": "assistant"}
                if content:
                    assistant_message["content"] = content
                if ordered_tool_calls:
                    assistant_message["tool_calls"] = ordered_tool_calls

                if ordered_tool_calls:
                    model_responses = [
                        {tool_call["function"]["name"]: tool_call["function"]["arguments"]}
                        for tool_call in ordered_tool_calls
                    ]
                else:
                    model_responses = content

                timing = {
                    "turn": state["turn_idx"],
                    "step": state["step_idx"],
                    "classification": classification,
                    "prompt_tokens": prompt_tokens,
                    "tool_schema_tokens": tool_schema_tokens,
                    "output_tokens": output_tokens,
                    "ttft_ms": round(ttft_ms, 2),
                    "total_ms": round(total_ms, 2),
                    "decode_ms": round(max(total_ms - ttft_ms, 0.0), 2),
                    "tool_calls_in_response": len(ordered_tool_calls),
                }
                if attempt_idx:
                    timing["stream_retries"] = attempt_idx

                response = {
                    "model_responses": model_responses,
                    "model_responses_message_for_chat_history": assistant_message,
                    "tool_call_ids": [tool_call["id"] for tool_call in ordered_tool_calls],
                    "input_token": input_tokens,
                    "output_token": output_tokens,
                    "assistant_content": content,
                }
                return response, timing
            except BadRequestError as exc:
                message = str(exc)
                overflow = "maximum context length" in message or (
                    "requested" in message and "tokens in the messages" in message
                )
                if not overflow:
                    raise
                if stream is not None:
                    close = getattr(stream, "close", None)
                    if callable(close):
                        close()

                before_messages = self._normalize_messages(state["messages"])
                before_prompt_tokens = self.prompt_tokens(split_multi_tool_calls(before_messages), tools)
                self._maybe_compress(state)
                after_messages = self._normalize_messages(state["messages"])
                if after_messages == before_messages:
                    raise

                messages = split_multi_tool_calls(after_messages)
                classification = self._classify_step(state)
                prompt_tokens = self.prompt_tokens(messages, tools)
                tool_schema_tokens = self.tools_tokens(tools)
                request["messages"] = messages
                self._append_prompt_log(
                    sample_id,
                    {
                        "type": "query_retry_after_compression",
                        "turn": state["turn_idx"],
                        "step": state["step_idx"],
                        "classification": classification,
                        "pre_prompt_tokens": before_prompt_tokens,
                        "prompt_tokens": prompt_tokens,
                        "tool_schema_tokens": tool_schema_tokens,
                        "messages": messages,
                        "tools": tools,
                        "reason": message,
                    },
                )
                continue
            except Exception as exc:
                retryable = _is_retryable_stream_error(exc)
                if not retryable:
                    raise
                if stream is not None:
                    close = getattr(stream, "close", None)
                    if callable(close):
                        close()
                if attempt_idx >= max_attempts - 1:
                    request_non_stream = dict(request)
                    request_non_stream["stream"] = False

                    t0 = time.perf_counter()
                    completion = self.client.chat.completions.create(**request_non_stream)
                    total_ms = (time.perf_counter() - t0) * 1000.0

                    choice = completion.choices[0]
                    message = choice.message
                    message_content = getattr(message, "content", "")
                    if message_content is None:
                        content = ""
                    elif isinstance(message_content, str):
                        content = message_content
                    else:
                        content = json.dumps(message_content, ensure_ascii=False)

                    ordered_tool_calls = []
                    for tool_call in getattr(message, "tool_calls", []) or []:
                        if hasattr(tool_call, "model_dump"):
                            ordered_tool_calls.append(tool_call.model_dump(exclude_none=True))
                        else:
                            function = getattr(tool_call, "function", None)
                            ordered_tool_calls.append(
                                {
                                    "id": getattr(tool_call, "id", ""),
                                    "type": getattr(tool_call, "type", "function") or "function",
                                    "function": {
                                        "name": getattr(function, "name", "") or "",
                                        "arguments": getattr(function, "arguments", "") or "",
                                    },
                                }
                            )

                    output_token_fallback_text = content + json.dumps(ordered_tool_calls, ensure_ascii=False)
                    output_tokens = len(self.tokenizer.tokenize(output_token_fallback_text)) if output_token_fallback_text else 0
                    usage = getattr(completion, "usage", None)
                    input_tokens = prompt_tokens
                    if usage is not None:
                        input_tokens = getattr(usage, "prompt_tokens", prompt_tokens) or prompt_tokens
                        output_tokens = getattr(usage, "completion_tokens", output_tokens) or output_tokens

                    assistant_message = {"role": "assistant"}
                    if content:
                        assistant_message["content"] = content
                    if ordered_tool_calls:
                        assistant_message["tool_calls"] = ordered_tool_calls

                    if ordered_tool_calls:
                        model_responses = [
                            {tool_call["function"]["name"]: tool_call["function"]["arguments"]}
                            for tool_call in ordered_tool_calls
                        ]
                    else:
                        model_responses = content

                    timing = {
                        "turn": state["turn_idx"],
                        "step": state["step_idx"],
                        "classification": classification,
                        "prompt_tokens": prompt_tokens,
                        "tool_schema_tokens": tool_schema_tokens,
                        "output_tokens": output_tokens,
                        "ttft_ms": round(total_ms, 2),
                        "total_ms": round(total_ms, 2),
                        "decode_ms": 0.0,
                        "tool_calls_in_response": len(ordered_tool_calls),
                        "stream_retries": attempt_idx + 1,
                        "transport_fallback": "non_stream",
                    }
                    response = {
                        "model_responses": model_responses,
                        "model_responses_message_for_chat_history": assistant_message,
                        "tool_call_ids": [tool_call["id"] for tool_call in ordered_tool_calls],
                        "input_token": input_tokens,
                        "output_token": output_tokens,
                        "assistant_content": content,
                    }
                    return response, timing
                if backoff_s > 0:
                    time.sleep(backoff_s)

    def _decode_execute(self, model_responses):
        if not isinstance(model_responses, list):
            return []
        coerced = _coerce_tool_call_args(model_responses)
        return _convert_to_function_call_coerced(coerced)

    def _prepare_market_price_execution(
        self,
        response: dict,
        initial_config: dict,
        involved_classes: list[str],
        sample_id: str,
        long_context: bool,
        turn_idx: int,
        step_idx: int,
    ) -> dict | None:
        model_responses = response.get("model_responses")
        if not isinstance(model_responses, list) or len(model_responses) != 1:
            return None

        coerced = _coerce_tool_call_args(model_responses)
        if len(coerced) != 1:
            return None

        func_name, args = next(iter(coerced[0].items()))
        if func_name != "place_order" or not isinstance(args, dict):
            return None

        raw_price = args.get("price")
        if not _looks_like_market_price(raw_price):
            return None

        symbol = args.get("symbol")
        if not isinstance(symbol, str) or not symbol:
            return None

        assistant_message = deepcopy(response["model_responses_message_for_chat_history"])
        original_tool_calls = assistant_message.get("tool_calls") or []
        if len(original_tool_calls) != 1:
            return None

        original_tool_call = deepcopy(original_tool_calls[0])
        original_tool_call_id = original_tool_call.get("id") or f"call_{turn_idx}_{step_idx}_place_order"
        lookup_tool_call_id = f"{original_tool_call_id}_price_lookup"

        lookup_calls = _convert_to_function_call_coerced([
            {"get_stock_info": {"symbol": symbol}},
        ])
        lookup_results, _ = execute_multi_turn_func_call(
            lookup_calls,
            initial_config,
            involved_classes,
            self.model_name_underline_replaced,
            sample_id,
            long_context=long_context,
            is_evaL_run=False,
        )
        if not lookup_results:
            return None

        lookup_result = lookup_results[0]
        resolved_price = _extract_price_from_stock_info(lookup_result)
        if resolved_price is None:
            return None

        resolved_args = dict(args)
        resolved_args["price"] = resolved_price
        resolved_calls = _convert_to_function_call_coerced(
            [
                {"get_stock_info": {"symbol": symbol}},
                {"place_order": resolved_args},
            ]
        )
        place_order_results, _ = execute_multi_turn_func_call(
            [resolved_calls[1]],
            initial_config,
            involved_classes,
            self.model_name_underline_replaced,
            sample_id,
            long_context=long_context,
            is_evaL_run=False,
        )
        if not place_order_results:
            return None

        lookup_tool_call = {
            "id": lookup_tool_call_id,
            "type": "function",
            "function": {
                "name": "get_stock_info",
                "arguments": json.dumps({"symbol": symbol}, ensure_ascii=False),
            },
        }
        resolved_place_order_tool_call = deepcopy(original_tool_call)
        resolved_place_order_tool_call["id"] = original_tool_call_id
        resolved_place_order_tool_call["function"] = {
            "name": "place_order",
            "arguments": json.dumps(resolved_args, ensure_ascii=False),
        }
        assistant_message["tool_calls"] = [lookup_tool_call, resolved_place_order_tool_call]
        if assistant_message.get("content") is None:
            assistant_message.pop("content", None)

        return {
            "assistant_message": assistant_message,
            "decoded_calls": resolved_calls,
            "execution_results": [lookup_result, place_order_results[0]],
            "tool_call_ids": [lookup_tool_call_id, original_tool_call_id],
            "meta": {
                "strategy": "inject_get_stock_info_before_place_order",
                "symbol": symbol,
                "original_price": raw_price,
                "resolved_price": resolved_price,
                "lookup_call": resolved_calls[0],
                "place_order_call": resolved_calls[1],
            },
        }

    def _append_tool_results(self, state: dict, execution_results: list[str], tool_call_ids: list[str]) -> None:
        for execution_result, tool_call_id in zip(execution_results, tool_call_ids):
            state["messages"].append(
                {
                    "role": "tool",
                    "content": execution_result,
                    "tool_call_id": tool_call_id,
                }
            )

    def _init_state(self, test_entry: dict, checkpoint: dict | None) -> dict:
        if checkpoint is not None:
            state = checkpoint
            state["resume_count"] = state.get("resume_count", 0) + 1
            state.setdefault("official_model_result", [])
            return state
        return {
            "sample_id": test_entry["id"],
            "turn_idx": 0,
            "step_idx": 0,
            "messages": [],
            "tools": self._build_tools(test_entry),
            "timing_log": [],
            "abc_snapshots": [],
            "prompt_snapshots": [],
            "turn_traces": [],
            "official_model_result": [],
            "executed_calls_history": [],
            "cumulative_tool_calls": 0,
            "previous_summary": None,
            "last_was_compression": False,
            "total_turns": len(test_entry.get("question", [])),
            "resume_count": 0,
        }

    def run_sample(self, test_entry: dict) -> dict:
        sample_id = test_entry["id"]
        checkpoint = self.load_checkpoint(sample_id)
        state = self._init_state(test_entry, checkpoint)
        self._rebuild_tool_state(test_entry, state)

        initial_config = test_entry.get("initial_config", {})
        involved_classes = test_entry["involved_classes"]
        test_category = sample_id.rsplit("_", 1)[0]
        long_context = "long_context" in test_category or "composite" in test_category
        holdout_function = test_entry.get("missed_function", {})

        if checkpoint is None:
            prompt_log = self.dirs["prompt_logs"] / f"{sample_id}.jsonl"
            if prompt_log.exists():
                prompt_log.unlink()

        while state["turn_idx"] < state["total_turns"]:
            turn_idx = state["turn_idx"]
            current_turn_message = deepcopy(test_entry["question"][turn_idx])

            if str(turn_idx) in holdout_function:
                test_entry["function"].extend(holdout_function[str(turn_idx)])
                state["tools"] = self._build_tools(test_entry)
                current_turn_message = [
                    {
                        "role": "user",
                        "content": DEFAULT_USER_PROMPT_FOR_ADDITIONAL_FUNCTION_FC,
                    }
                ]

            if state["step_idx"] == 0:
                state["messages"].extend(current_turn_message)
                if turn_idx > 0:
                    self._maybe_compress(state)
                while len(state["official_model_result"]) <= turn_idx:
                    state["official_model_result"].append([])
                state["turn_traces"].append(
                    {
                        "turn": turn_idx,
                        "user_messages": current_turn_message,
                        "steps": [],
                    }
                )

            while True:
                response, timing = self._stream_query(state)
                state["official_model_result"][turn_idx].append(deepcopy(response["model_responses"]))
                state["timing_log"].append(timing)
                prepared_execution = self._prepare_market_price_execution(
                    response,
                    initial_config,
                    involved_classes,
                    sample_id,
                    long_context,
                    state["turn_idx"],
                    state["step_idx"],
                )
                if prepared_execution is not None:
                    response["model_responses_message_for_chat_history"] = prepared_execution["assistant_message"]
                state["messages"].append(response["model_responses_message_for_chat_history"])

                step_trace = {
                    "turn": state["turn_idx"],
                    "step": state["step_idx"],
                    "classification": timing["classification"],
                    "prompt_tokens": timing["prompt_tokens"],
                    "ttft_ms": timing["ttft_ms"],
                    "total_ms": timing["total_ms"],
                    "decode_ms": timing["decode_ms"],
                    "tool_calls_in_response": timing["tool_calls_in_response"],
                    "model_responses": response["model_responses"],
                    "assistant_content": response.get("assistant_content", ""),
                }
                if prepared_execution is not None:
                    step_trace["market_price_resolution"] = prepared_execution["meta"]

                try:
                    if prepared_execution is not None:
                        decoded_calls = prepared_execution["decoded_calls"]
                    else:
                        decoded_calls = self._decode_execute(response["model_responses"])
                    step_trace["decoded_calls"] = decoded_calls
                except Exception as exc:
                    decoded_calls = []
                    step_trace["decode_error"] = str(exc)

                should_end_turn = False
                if is_empty_execute_response(decoded_calls):
                    should_end_turn = True
                    step_trace["end_reason"] = "empty_or_text_response"

                if not should_end_turn:
                    if prepared_execution is not None:
                        execution_results = prepared_execution["execution_results"]
                        tool_call_ids = prepared_execution["tool_call_ids"]
                    else:
                        execution_results, _ = execute_multi_turn_func_call(
                            decoded_calls,
                            initial_config,
                            involved_classes,
                            self.model_name_underline_replaced,
                            sample_id,
                            long_context=long_context,
                            is_evaL_run=False,
                        )
                        tool_call_ids = response["tool_call_ids"]
                    self._append_tool_results(state, execution_results, tool_call_ids)
                    state["cumulative_tool_calls"] += len(decoded_calls)
                    state["executed_calls_history"].append(decoded_calls)
                    step_trace["execution_results"] = execution_results

                state["turn_traces"][turn_idx]["steps"].append(step_trace)

                if state.get("last_was_compression"):
                    state["last_was_compression"] = False

                if should_end_turn:
                    state["turn_idx"] += 1
                    state["step_idx"] = 0
                    self._save_checkpoint(state, test_entry)
                    self._persist_sample_logs(state)
                    break

                state["step_idx"] += 1
                if state["step_idx"] > MAXIMUM_STEP_LIMIT:
                    step_trace["end_reason"] = "maximum_step_limit"
                    state["turn_idx"] = state["total_turns"]
                    state["step_idx"] = 0
                    self._save_checkpoint(state, test_entry)
                    self._persist_sample_logs(state)
                    raise RuntimeError(f"{sample_id} exceeded maximum step limit")

                self._save_checkpoint(state, test_entry)
                self._persist_sample_logs(state)

        self._persist_sample_logs(state)
        self.clear_checkpoint(sample_id)
        return {
            "id": sample_id,
            "steps": len(state["timing_log"]),
            "compressions": len(state["abc_snapshots"]),
            "tool_calls": state["cumulative_tool_calls"],
            "resume_count": state.get("resume_count", 0),
        }
