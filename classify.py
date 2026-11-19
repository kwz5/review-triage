"""Classify reviews with Gemini and save every result to SQLite.

Usage:
    python classify.py data/labeling_set.csv        # your 50 labeled reviews
    python classify.py data/sample.csv              # all 500
    python classify.py data/sample.csv --limit 5    # quick test

Safe to stop and re-run: reviews already done for the current prompt_version
are skipped. Change prompt_version in config.yaml to start a fresh run.
"""
import argparse
import json
import sqlite3
import time
from datetime import datetime, timezone

import pandas as pd
import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

STOP_CODES = {400, 401, 403, 404}  # config problems: retrying will not help

SCHEMA = """
CREATE TABLE IF NOT EXISTS results (
    prompt_version TEXT,
    id INTEGER,
    review TEXT,
    rating INTEGER,
    thumbsup INTEGER,
    submitted TEXT,
    primary_theme TEXT,
    secondary_theme TEXT,
    sentiment TEXT,
    severity INTEGER,
    reason TEXT,
    status TEXT,
    raw_response TEXT,
    model TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    created_at TEXT,
    PRIMARY KEY (prompt_version, id)
)
"""


def load_config(path="config.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_prompt(cfg, review_text):
    cats = "\n".join(f"- {name}: {desc}" for name, desc in cfg["categories"].items())
    rules = "\n".join(f"- {r}" for r in cfg["rules"])
    return f'''You are classifying app-store reviews for a product team.

Themes (choose exactly one primary_theme, and optionally one secondary_theme):
{cats}

Severity of the PROBLEM described (not the sentiment, and ignore the star rating):
{cfg["severity_scale"]}

Rules:
{rules}

Respond with ONLY a JSON object with these keys:
"primary_theme": one theme name from the list,
"secondary_theme": one theme name from the list, or null,
"sentiment": "positive", "neutral" or "negative",
"severity": an integer from 0 to 5,
"reason": a short explanation of 15 words or fewer

Review:
"""{review_text}"""'''


def call_model(client, cfg, prompt):
    """Call the API, retrying temporary errors with growing waits."""
    gen_config = types.GenerateContentConfig(
        temperature=0, response_mime_type="application/json"
    )
    for attempt in range(cfg["max_retries"]):
        try:
            return client.models.generate_content(
                model=cfg["model"], contents=prompt, config=gen_config
            )
        except Exception as e:
            code = getattr(e, "code", None)
            if code in STOP_CODES:
                raise  # wrong key, wrong model name, etc. Stop and show the error.
            wait = 2 * 2**attempt
            print(f"    API error ({code}). Retrying in {wait}s...")
            time.sleep(wait)
    return None  # gave up on this review


def parse_result(raw, cfg):
    """Turn the model's JSON text into a clean, validated dict."""
    text = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    data = json.loads(text)
    valid = set(cfg["categories"])

    primary = data.get("primary_theme")
    if primary not in valid:
        raise ValueError(f"unknown primary_theme: {primary!r}")
    secondary = data.get("secondary_theme")
    if secondary not in valid or secondary == primary:
        secondary = None

    severity = int(data.get("severity"))
    if not 0 <= severity <= 5:
        raise ValueError(f"severity out of range: {severity}")
    if primary == "praise_no_issue":
        severity = 0

    return {
        "primary_theme": primary,
        "secondary_theme": secondary,
        "sentiment": data.get("sentiment"),
        "severity": severity,
        "reason": data.get("reason"),
    }


def clean(value, cast=None):
    """Convert pandas/numpy values into plain Python values SQLite accepts."""
    if value is None or pd.isna(value):
        return None
    return cast(value) if cast else value


def save(con, record):
    cols = ", ".join(record)
    marks = ", ".join("?" for _ in record)
    con.execute(f"INSERT OR REPLACE INTO results ({cols}) VALUES ({marks})", list(record.values()))
    con.commit()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="CSV of reviews to classify")
    ap.add_argument("--limit", type=int, help="only do the first N rows")
    args = ap.parse_args()

    cfg = load_config()
    version = cfg["prompt_version"]
    cols = cfg["columns"]

    df = pd.read_csv(args.csv)
    if args.limit:
        df = df.head(args.limit)

    con = sqlite3.connect(cfg["database"])
    con.execute(SCHEMA)
    done = {
        r[0]
        for r in con.execute(
            "SELECT id FROM results WHERE prompt_version = ? AND status = 'ok'", (version,)
        )
    }
    todo = df[~df[cols["id"]].isin(done)]
    print(f"Prompt {version}, model {cfg['model']}: {len(todo)} to do, {len(done)} already done")

    client = genai.Client()
    api_failures_in_a_row = 0

    for n, (_, row) in enumerate(todo.iterrows(), 1):
        rid = int(row[cols["id"]])
        text = str(row[cols["text"]])
        record = {
            "prompt_version": version,
            "id": rid,
            "review": text,
            "rating": clean(row.get(cols.get("rating")), int),
            "thumbsup": clean(row.get(cols.get("thumbsup")), int),
            "submitted": clean(row.get(cols.get("submitted")), str),
            "model": cfg["model"],
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }

        response = call_model(client, cfg, build_prompt(cfg, text))
        if response is None:
            record["status"] = "api_failed"
            api_failures_in_a_row += 1
        else:
            api_failures_in_a_row = 0
            usage = getattr(response, "usage_metadata", None)
            record["input_tokens"] = getattr(usage, "prompt_token_count", None)
            record["output_tokens"] = getattr(usage, "candidates_token_count", None)
            try:
                raw = response.text
            except Exception:
                raw = None
            record["raw_response"] = raw
            if not raw:
                record["status"] = "blank"  # e.g. blocked by a safety filter
            else:
                try:
                    record.update(parse_result(raw, cfg))
                    record["status"] = "ok"
                except (ValueError, TypeError, json.JSONDecodeError) as e:
                    record["status"] = "parse_error"
                    record["reason"] = str(e)

        save(con, record)
        print(f"[{n}/{len(todo)}] id {rid}: {record['status']} "
              f"{record.get('primary_theme') or ''} {record.get('severity', '')}")

        if api_failures_in_a_row >= 5:
            print("5 API failures in a row. Stopping. Wait a bit and re-run; progress is saved.")
            break
        time.sleep(cfg["delay_seconds"])

    summary = con.execute(
        "SELECT status, COUNT(*) FROM results WHERE prompt_version = ? GROUP BY status", (version,)
    ).fetchall()
    print("Done. Status counts:", dict(summary))
    con.close()


if __name__ == "__main__":
    main()