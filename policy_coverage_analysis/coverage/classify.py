#!/usr/bin/env python3
"""Label every scenario on the five classification axes with an LLM.

Provider is chosen from --model: a name starting with "gpt" or "o" goes to
OpenAI (Responses API, strict json_schema), anything else to Gemini
(generateContent, responseSchema). Both are pinned to temperature 0 and to
the SAME schema from coverage/axes.py, so the two paths differ only in the
transport. Switching model invalidates the cache automatically - digest()
hashes the model name - so a provider change never reuses old labels.

One request per STUDY -- the scenarios documented by the same paper go in
together, up to ``--batch`` of them -- at temperature 0 with a schema-constrained
JSON output.  Several of the axes ask whether a scenario DEPARTS from the
default, and a departure is only visible against its siblings: a technology
assumption every member of a study states is that study's baseline, while one
that separates two members is a treatment.  Labelling a description alone hides
that distinction, and the model can do no better than guess at it.

Answers are cached by SHA-256 of (prompt version, model, every scenario and
description in the batch).  The unit is therefore the batch, not the row:
editing one description re-labels its whole study.  That is the cost of the
comparison - the verdicts for its siblings were reached by reading it.

Every ``true`` must come with a span copied from the description.  The span is
checked against the description here; mismatches are counted and reported, never
silently accepted or silently dropped.
"""

from __future__ import annotations

import argparse
import hashlib
import threading
import http.client
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import xlsxwriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common import config, io
from coverage.axes import AXES, PROMPT_VERSION, RESPONSE_SCHEMA, build_prompt

HERE = Path(__file__).resolve().parent
DEFAULT_CACHE = HERE / "cache/scenario_labels.json"
DEFAULT_OUTPUT = HERE / "output/scenario_labels.csv"
DEFAULT_WORKBOOK = HERE / "output/scenario_labels.xlsx"
ENDPOINT = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
RETRY_STATUS = {408, 429, 500, 502, 503, 504}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="gpt-5.6")
    parser.add_argument("--descriptions", type=Path, default=config.DESCRIPTIONS)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--workbook", type=Path, default=DEFAULT_WORKBOOK)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--workers", type=int, default=8,
                        help="concurrent requests; lower it if the API rate-limits")
    parser.add_argument("--timeout", type=int, default=45,
                        help="seconds per request, PLUS this many again for every "
                             "10 scenarios in the batch; a 182-scenario request "
                             "writes 182 x 6 verdicts and needs minutes, not the "
                             "two a single description needed")
    parser.add_argument("--batch", type=int, default=config.BATCH,
                        help="max scenarios per request; a study larger than this "
                             "is split, and the split halves see less of each other")
    parser.add_argument("--reasoning", default="none",
                        choices=["none", "minimal", "low", "medium", "high", "default"],
                        help="OpenAI reasoning effort; 'none' turns thinking off, "
                             "'default' omits the parameter and lets the model "
                             "decide. Ignored on the Gemini path")
    parser.add_argument("--limit", type=int, help="only label the first N scenarios")
    return parser.parse_args(argv)


def digest(model: str, items: list[tuple[str, str]], reasoning: str = "none") -> str:
    """One key per batch: the prompt the batch produces is what was paid for.

    Reasoning effort is part of the key because it changes the answer, not just
    the price: labels found with thinking off are not the same labels.
    """
    payload = "\0".join([PROMPT_VERSION, model, reasoning]
                        + [f"{name}\x1f{text}" for name, text in items])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def batches(groups: list[list[str]], text: dict[str, str],
            size: int) -> list[list[tuple[str, str]]]:
    """Studies, in order, each split into chunks of at most ``size``."""
    out: list[list[tuple[str, str]]] = []
    for members in groups:
        pairs = [(name, text[name]) for name in members if name in text]
        for start in range(0, len(pairs), size):
            out.append(pairs[start:start + size])
    return out


def load_cache(path: Path) -> dict[str, dict]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def save_cache(path: Path, cache: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".json.tmp")
    temp.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def uses_openai(model: str) -> bool:
    """OpenAI model names start with gpt- or o<digit>; everything else is Gemini."""
    m = model.strip().lower()
    return m.startswith("gpt") or (len(m) > 1 and m[0] == "o" and m[1].isdigit())


def api_key_name(model: str) -> str:
    return "OPENAI_API_KEY" if uses_openai(model) else "GEMINI_API_KEY"


def _strict_schema(node):
    """OpenAI strict json_schema requires additionalProperties:false on every
    object. Gemini rejects nothing, but does not need it, so the shared schema in
    axes.py stays provider-neutral and the flag is added here."""
    if isinstance(node, dict):
        out = {k: _strict_schema(v) for k, v in node.items()}
        if out.get("type") == "object":
            out["additionalProperties"] = False
        return out
    if isinstance(node, list):
        return [_strict_schema(v) for v in node]
    return node


def call_openai(items: list[tuple[str, str]], model: str, api_key: str,
                retries: int, timeout: int = 45,
                reasoning: str = "none") -> dict[str, dict]:
    """OpenAI path. Same contract as call_gemini below."""
    from openai import OpenAI
    client = OpenAI(api_key=api_key)
    window = timeout + timeout * len(items) // 10
    wanted = {name for name, _ in items}
    # "default" means send nothing and let the model pick; any other value is
    # passed through, and "none" is what turns thinking off.
    extra = {} if reasoning == "default" else {"reasoning": {"effort": reasoning}}
    for attempt in range(retries):
        try:
            resp = client.responses.create(
                model=model,
                temperature=0,
                max_output_tokens=65536,
                input=[{"role": "user",
                        "content": [{"type": "input_text", "text": build_prompt(items)}]}],
                text={"format": {"type": "json_schema", "name": "scenario_labels",
                                 "schema": _strict_schema(RESPONSE_SCHEMA),
                                 "strict": True}},
                timeout=window,
                **extra,
            )
            results = json.loads(resp.output_text).get("results") or []
            return {r["scenario"]: r for r in results
                    if isinstance(r, dict) and r.get("scenario") in wanted}
        except Exception as exc:                       # noqa: BLE001
            detail = str(exc)
            # A rejected parameter will not fix itself on retry - fail loudly so
            # the model choice can be corrected instead of burning the retries.
            if any(k in detail.lower() for k in
                   ("unsupported parameter", "unknown parameter", "temperature",
                    "max_output_tokens", "json_schema", "invalid_request",
                    "reasoning")):
                raise RuntimeError(f"OpenAI rejected the request (model={model}): "
                                   f"{detail[:500]}") from exc
            if attempt + 1 >= retries:
                raise RuntimeError(f"OpenAI request failed: {detail[:500]}") from exc
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def call_gemini(items: list[tuple[str, str]], model: str, api_key: str,
                retries: int, timeout: int = 45) -> dict[str, dict]:
    """Label one batch; return {scenario: answer}, keyed as the batch named them.

    A model that drops a scenario from its results is a real failure mode at
    this size, so the caller checks what came back rather than trusting the
    count.  Names are matched exactly - they were given exactly.
    """
    body = json.dumps(
        {
            "contents": [{"parts": [{"text": build_prompt(items)}]}],
            "generationConfig": {
                "temperature": 0,
                "responseMimeType": "application/json",
                "responseSchema": RESPONSE_SCHEMA,
                # A schema-constrained answer that runs out of output budget is
                # truncated JSON, which arrives here as a parse error rather than
                # as a short answer.  Ask for room the largest batch can use.
                "maxOutputTokens": 65536,
            },
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        ENDPOINT.format(model=model),
        data=body,
        method="POST",
        headers={"x-goog-api-key": api_key, "Content-Type": "application/json"},
    )
    # Time scales with the batch: the model writes one verdict and one quoted
    # span per axis per scenario, so a 182-scenario request is minutes of output.
    window = timeout + timeout * len(items) // 10
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(request, timeout=window) as response:
                payload = json.loads(response.read().decode("utf-8"))
            text = payload["candidates"][0]["content"]["parts"][0]["text"]
            results = json.loads(text).get("results") or []
            wanted = {name for name, _ in items}
            return {r["scenario"]: r for r in results
                    if isinstance(r, dict) and r.get("scenario") in wanted}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:500]
            if attempt + 1 >= retries or exc.code not in RETRY_STATUS:
                raise RuntimeError(f"Gemini HTTP {exc.code}: {detail}") from exc
        except (urllib.error.URLError, ConnectionError, http.client.IncompleteRead,
                TimeoutError, json.JSONDecodeError) as exc:
            if attempt + 1 >= retries:
                raise RuntimeError(f"Gemini request failed: {exc}") from exc
        time.sleep(2**attempt)
    raise AssertionError("unreachable")


def write_workbook(path: Path, labels: pd.DataFrame, descriptions: pd.DataFrame,
                   model: str) -> None:
    """One reviewable row per scenario: text, every verdict, every cited span."""
    table = descriptions.merge(labels, on="scenario", how="left")
    table.insert(2, "n_axes_true", table[[axis.key for axis in AXES]].sum(axis=1))

    path.parent.mkdir(parents=True, exist_ok=True)
    book = xlsxwriter.Workbook(path)
    wrap = book.add_format({"text_wrap": True, "valign": "top", "border": 1})
    plain = book.add_format({"valign": "top", "border": 1})
    centre = book.add_format({"valign": "top", "border": 1, "align": "center"})
    header = book.add_format({"bold": True, "bg_color": "#EFEFEF", "border": 1,
                              "text_wrap": True, "valign": "top"})
    yes = book.add_format({"bg_color": "#D6EAD6", "bold": True, "align": "center",
                           "valign": "top", "border": 1})

    sheet = book.add_worksheet("Labels")
    columns = ["scenario", "description", "n_axes_true"]
    columns += [axis.key for axis in AXES]
    columns += [f"{axis.key}__evidence" for axis in AXES]
    sheet.write_row(0, 0, columns, header)
    widths = {"scenario": 30, "description": 90, "n_axes_true": 7}
    for index, name in enumerate(columns):
        if name in widths:
            style, width = (wrap if name == "description" else centre), widths[name]
        elif name.endswith("__evidence"):
            style, width = wrap, 55
        else:
            style, width = centre, 13
        sheet.set_column(index, index, width, style)
        sheet.write_column(1, index, table[name].tolist(), style)
    first_axis = columns.index(AXES[0].key)
    sheet.conditional_format(1, first_axis, len(table), first_axis + len(AXES) - 1,
                             {"type": "cell", "criteria": "==", "value": True, "format": yes})
    sheet.freeze_panes(1, 1)
    sheet.autofilter(0, 0, len(table), len(columns) - 1)

    notes = book.add_worksheet("Prompt")
    notes.set_column(0, 0, 20, plain)
    notes.set_column(1, 1, 130, wrap)
    rows = [("model", model), ("prompt_version", PROMPT_VERSION),
            ("scenarios", str(len(table)))]
    rows += [(f"{axis.key} [{axis.dimension}]", axis.question) for axis in AXES]
    rows.append(("full prompt", build_prompt(
        [("<SCENARIO 1>", "<DESCRIPTION 1>"), ("<SCENARIO 2>", "<DESCRIPTION 2>")])))
    for offset, (name, value) in enumerate(rows):
        notes.write(offset, 0, name, header if offset < 3 else plain)
        notes.write(offset, 1, value, wrap)
    book.close()


def label_batch(items: list[tuple[str, str]], model: str, api_key: str,
                retries: int, timeout: int = 45,
                reasoning: str = "none") -> dict[str, dict]:
    """One batch, then a second pass for whatever the model left out.

    A large batch sometimes comes back short - the model answers most of the
    list and stops.  Re-asking only the stragglers costs one small call and
    keeps the study intact; giving up would leave those scenarios unlabelled
    with nothing to show for it.
    """
    if uses_openai(model):
        def call(batch):
            return call_openai(batch, model, api_key, retries, timeout, reasoning)
    else:
        def call(batch):
            return call_gemini(batch, model, api_key, retries, timeout)
    answers = call(items)
    absent = [pair for pair in items if pair[0] not in answers]
    if absent and len(absent) < len(items):
        answers |= call(absent)
    return answers


def fetch_missing(missing: list[tuple[str, list[tuple[str, str]]]],
                  cache: dict[str, dict], cache_path: Path,
                  args: argparse.Namespace, api_key: str) -> None:
    """Fill the cache for every uncached batch, several requests at a time.

    The cache is shared mutable state, so writes take a lock, and it is flushed
    after every batch: one batch can be 200 scenarios, and a run interrupted
    halfway should keep what it has already paid for.
    """
    lock = threading.Lock()
    done = 0
    short: list[str] = []
    failed: list[tuple[str, int, str]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(label_batch, items, args.model, api_key, args.retries,
                        args.timeout, args.reasoning): (key, items)
            for key, items in missing
        }
        try:
            for future in as_completed(futures):
                key, items = futures[future]
                try:
                    answers = future.result()
                except Exception as exc:            # noqa: BLE001 - reported below
                    # One batch failing must not throw away the others: they are
                    # already paid for, and a re-run asks only for what is still
                    # uncached.  Timeouts are the common case here - a request
                    # carrying 182 scenarios is minutes of output.
                    with lock:
                        done += 1
                        failed.append((items[0][0], len(items), str(exc)[:120]))
                        print(f"[{done}/{len(missing)}] {len(items)} scenarios, "
                              f"{items[0][0]} ...  FAILED: {str(exc)[:90]}", flush=True)
                    continue
                with lock:
                    cache[key] = answers
                    done += 1
                    missed = len(items) - len(answers)
                    note = f"  MISSING {missed}" if missed else ""
                    if missed:
                        short.append(f"{items[0][0]} (+{len(items) - 1}): {missed}")
                    print(f"[{done}/{len(missing)}] {len(items)} scenarios, "
                          f"{items[0][0]} ...{note}", flush=True)
                    save_cache(cache_path, cache)
        finally:
            with lock:
                save_cache(cache_path, cache)
    if short:
        print(f"\nWARNING: {len(short)} batch(es) came back short even after a "
              f"second ask: {short[:3]}", flush=True)
    if failed:
        print(f"\nWARNING: {len(failed)} batch(es) failed and were not cached:",
              flush=True)
        for name, size, why in failed[:5]:
            print(f"    {name} (+{size - 1}): {why}", flush=True)
        print("  Re-run to ask for them again, or lower --batch / raise --timeout.",
              flush=True)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    descriptions = io.load_descriptions(args.descriptions)
    if args.limit:
        descriptions = descriptions.head(args.limit)
    api_key = io.read_api_key(config.ENV_FILE, api_key_name(args.model))
    cache = load_cache(args.cache)

    text = dict(zip(descriptions.scenario, descriptions.description))
    groups = io.load_groups(set(text), config.SCENARIO_REFERENCES, config.LITERATURE_IDS)
    work = batches(groups, text, args.batch)
    sizes = sorted((len(b) for b in work), reverse=True)
    print(f"{len(descriptions)} scenarios in {len(groups)} studies -> {len(work)} "
          f"request(s), largest {sizes[0] if sizes else 0}", flush=True)

    missing = [(digest(args.model, items, args.reasoning), items) for items in work
               if digest(args.model, items, args.reasoning) not in cache]
    print(f"{len(missing)} request(s) to make, {len(work) - len(missing)} cached",
          flush=True)
    if missing:
        fetch_missing(missing, cache, args.cache, args, api_key)

    answers: dict[str, dict] = {}
    unfetched: list[list[tuple[str, str]]] = []
    for items in work:
        key = digest(args.model, items, args.reasoning)
        if key in cache:
            answers |= cache[key]
        else:
            unfetched.append(items)
    if unfetched:
        # Whatever succeeded is cached, so a re-run asks only for these.
        listing = ", ".join(f"{b[0][0]} (+{len(b) - 1})" for b in unfetched[:5])
        raise SystemExit(
            f"{len(unfetched)} of {len(work)} request(s) did not complete: {listing}.\n"
            f"Everything else is cached. Re-run to retry just these; if they keep "
            f"timing out, raise --timeout (now {args.timeout}s per 10 scenarios) or "
            f"lower --batch (now {args.batch})."
        )
    # A batch that came back short is cached as it arrived - it was paid for -
    # so the stragglers are picked up here, in batches of their own, rather than
    # by discarding a partial answer and asking for the whole study again.
    absent = [name for name in text if name not in answers]
    if absent:
        print(f"\n{len(absent)} scenario(s) not covered by any batch; asking for "
              f"them on their own", flush=True)
        topup = batches([absent], text, args.batch)
        pending = [(digest(args.model, items, args.reasoning), items) for items in topup
                   if digest(args.model, items, args.reasoning) not in cache]
        if pending:
            fetch_missing(pending, cache, args.cache, args, api_key)
        for items in topup:
            answers |= cache[digest(args.model, items, args.reasoning)]
        absent = [name for name in text if name not in answers]
    if absent:
        raise ValueError(
            f"{len(absent)} scenario(s) were never labelled, e.g. {absent[:5]}"
        )

    rows: list[dict[str, object]] = []
    unverified: list[tuple[str, str]] = []
    for row in descriptions.itertuples(index=False):
        answer = answers[row.scenario]

        record: dict[str, object] = {"scenario": row.scenario}
        for axis in AXES:
            present = bool(answer[axis.key]["present"])
            evidence = answer[axis.key]["evidence"].strip()
            if present and not evidence:
                raise ValueError(f"{row.scenario}: {axis.key} is true with no evidence")
            if present and evidence.casefold() not in row.description.casefold():
                unverified.append((row.scenario, axis.key))
            record[axis.key] = present
            record[f"{axis.key}__evidence"] = evidence
        rows.append(record)

    labels = pd.DataFrame(rows)
    if args.limit:
        print(labels.to_string(index=False))
        print(f"\n--limit was set, so {args.output.name} was not written")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        labels.to_csv(args.output, index=False)
        write_workbook(args.workbook, labels, descriptions, args.model)

    counts = {axis.key: int(labels[axis.key].sum()) for axis in AXES}
    print(f"\n{len(labels)} scenarios labelled with {args.model}")
    for axis in AXES:
        print(f"  {counts[axis.key]:>4}  {axis.key}")
    print(f"  {int((~labels[[a.key for a in AXES]].any(axis=1)).sum()):>4}  no axis at all")
    if unverified:
        print(
            f"\nWARNING: {len(unverified)} labels cite a span that is not verbatim in the "
            f"description (e.g. {unverified[:3]}); the labels were kept, review them."
        )
    if not args.limit:
        print(f"\nSaved {args.output}\n      {args.workbook}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
