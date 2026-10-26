
# -*- coding: utf-8 -*-
"""
fewshot_codegen.py

Thai Query -> Python Code
ใช้ Few-shot Prompting กับ Ollama โดยรับคำสั่งภาษาไทย
และให้โมเดลตอบเป็น Python code เท่านั้น

Usage:
    python fewshot_codegen.py
    python fewshot_codegen.py --model qwen2.5-coder:3b
    python fewshot_codegen.py --no-fewshot
    python fewshot_codegen.py --tasks tasks.json --out results.json

    Default dataset: tasks_th.json (Thai queries). Point --tasks to
    another JSON file to swap datasets in the future.

Dependencies:
    pip install requests
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


# ============================================================
# Configuration
# ============================================================

OLLAMA_CHAT_URL = "http://localhost:11434/api/chat"
DEFAULT_MODEL = "qwen2.5-coder:3b"
DEFAULT_NUM_CTX = 8192

# Default dataset of Thai tasks. Kept in a separate JSON file so the
# dataset can be swapped without touching this script
# (override with --tasks / DEFAULT_TASKS_PATH).
DEFAULT_TASKS_PATH = Path(__file__).resolve().parent / "tasks_th.json"


SYSTEM_PROMPT = """
คุณเป็นผู้ช่วยเขียนโปรแกรม Python จากคำสั่งภาษาไทย
สร้างโค้ดที่ทำงานตรงตามความต้องการของโจทย์ล่าสุด

รูปแบบคำตอบ:
ส่งเฉพาะโค้ด Python ที่สมบูรณ์
ไม่ใส่ Markdown, code fence, หัวข้อ หรือคำอธิบายนอกโค้ดในภาษาใด
ไม่ส่ง JSON เพื่อครอบคำตอบ

กฎการเขียนโค้ด:
1. รักษาความหมายและเงื่อนไขทั้งหมดของโจทย์ล่าสุด
   ตัวอย่างก่อนหน้าใช้เป็นแนวทางเท่านั้น
   ห้ามนำเงื่อนไขหรือข้อจำกัดของตัวอย่างอื่นมาแทนโจทย์ล่าสุด

2. ถ้าผู้ใช้ระบุชื่อฟังก์ชัน คลาส ตัวแปร หรือพารามิเตอร์
   ให้ใช้ชื่อตามที่กำหนด
   ถ้าไม่ระบุ ให้เลือกชื่อที่สื่อความหมายและเหมาะสม

3. ใช้ Python standard library ก่อน
   เว้นแต่โจทย์จำเป็นต้องใช้ไลบรารีอื่น
   ใส่ imports ที่จำเป็นให้ครบ

4. ถ้าโจทย์ขอฟังก์ชันหรือคลาส
   ให้ส่งนิยามที่สมบูรณ์และ imports ที่จำเป็น
   ไม่เพิ่มตัวอย่างการเรียกใช้งาน โค้ดทดสอบ หรือ main
   เว้นแต่ผู้ใช้ร้องขอ

5. ถ้าโจทย์ระบุให้ผู้ใช้กรอกข้อมูลและแสดงผล
   ให้มีส่วนรับข้อมูลและแสดงผลตามที่ระบุ
   อย่าใช้ข้อมูลตัวอย่างแบบตายตัวแทนการรับข้อมูลจริง

6. รักษาชนิดผลลัพธ์ ลำดับข้อมูล และพฤติกรรมเมื่อเกิดข้อผิดพลาด
   ตามที่โจทย์กำหนด
   ไม่เพิ่มความสามารถหรือข้อห้ามที่ผู้ใช้ไม่ได้ร้องขอ

7. คำนึงถึงข้อมูลว่าง ข้อมูลซ้ำ และกรณีขอบที่เกี่ยวข้อง
   หากต้องสมมติโครงสร้างข้อมูล ให้สมมติเท่าที่จำเป็น

8. ทุกบล็อกคำสั่งต้องมีคำสั่ง Python ที่ถูกต้อง
   ห้ามส่งบล็อกที่มีเพียงคอมเมนต์
   ห้ามใช้ placeholder แทนส่วนการทำงานที่โจทย์ต้องการ

9. ก่อนส่งคำตอบ ให้ตรวจเครื่องหมายอัญประกาศ วงเล็บ
   เครื่องหมายโคลอน การเยื้องบรรทัด และความครบถ้วนของโค้ด
   แล้วส่งเฉพาะโค้ดสุดท้าย ไม่แสดงขั้นตอนการตรวจ

10. ห้ามอ้างว่ารันหรือทดสอบโค้ดแล้ว หากยังไม่ได้รันจริง
""".strip()

# ============================================================
# Few-shot examples
# Input  = Thai query
# Output = Python code only
# ============================================================

FEWSHOT_EXAMPLES: list[dict[str, str]] = [
    {
        "instruction": (
            "เขียนฟังก์ชัน fibonacci(n) เพื่อหาตัวเลขฟีโบนักชี "
            "ตำแหน่งที่ n โดยเริ่มนับจากตำแหน่ง 0 "
            "และใช้การวนซ้ำแทน recursion"
        ),
        "output": '''def fibonacci(n: int) -> int:
    if n < 0:
        raise ValueError("n must be non-negative")

    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b

    return a'''
    },
    {
        "instruction": (
            "เขียนฟังก์ชัน group_by(items, key_fn) "
            "เพื่อจัดกลุ่มข้อมูลตามค่าที่ได้จาก key_fn "
            "และคืนผลลัพธ์เป็น dictionary "
            "ที่แต่ละ key เก็บรายการข้อมูลไว้ใน list"
        ),
        "output": '''from collections import defaultdict


def group_by(items, key_fn):
    groups = defaultdict(list)

    for item in items:
        key = key_fn(item)
        groups[key].append(item)

    return dict(groups)'''
    },
    {
        "instruction": (
            "เขียนฟังก์ชัน parse_kv(text) "
            "เพื่อแปลงข้อความหลายบรรทัดที่มีรูปแบบ key=value "
            "เป็น dictionary โดยข้ามบรรทัดว่าง "
            "บรรทัดที่ขึ้นต้นด้วย # และบรรทัดที่ไม่มีเครื่องหมาย ="
        ),
        "output": '''def parse_kv(text: str) -> dict:
    result = {}

    for line in text.splitlines():
        line = line.strip()

        if not line or line.startswith("#"):
            continue

        if "=" not in line:
            continue

        key, _, value = line.partition("=")
        result[key.strip()] = value.strip()

    return result'''
    },
]


# ============================================================
# Target tasks are stored in tasks_th.json (see DEFAULT_TASKS_PATH).
# All queries are in Thai. Swap datasets by pointing --tasks to
# another JSON file with the same schema (id + instruction).
# ============================================================


# ============================================================
# Build messages
# ============================================================

def build_messages(
    instruction: str,
    fewshot: bool = False,
) -> list[dict[str, str]]:

    if not instruction.strip():
        raise ValueError("instruction must not be empty")

    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        }
    ]

    if fewshot:
        for example in FEWSHOT_EXAMPLES:
            messages.append({
                "role": "user",
                "content": example["instruction"],
            })

            messages.append({
                "role": "assistant",
                "content": example["output"],
            })

    messages.append({
        "role": "user",
        "content": instruction,
    })

    return messages

# ============================================================
# Call Ollama
# ============================================================

def call_ollama(
    messages: list[dict[str, str]],
    model: str,
    temperature: float = 0.0,
    seed: int | None = 42,
    num_ctx: int = DEFAULT_NUM_CTX,
    timeout: int = 300,
) -> dict[str, Any]:

    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_ctx": num_ctx,
        },
    }

    if seed is not None:
        payload["options"]["seed"] = seed

    start = time.perf_counter()

    response = requests.post(
        OLLAMA_CHAT_URL,
        json=payload,
        timeout=timeout,
    )
    response.raise_for_status()

    elapsed = time.perf_counter() - start
    data = response.json()

    eval_count = data.get("eval_count")
    eval_duration = data.get("eval_duration")

    return {
        "content": data.get("message", {}).get("content", ""),
        "total_seconds": round(elapsed, 3),
        "eval_count": eval_count,
        "prompt_eval_count": data.get("prompt_eval_count"),
        "tokens_per_second": (
            round(eval_count / (eval_duration / 1e9), 2)
            if eval_count and eval_duration
            else None
        ),
    }


# ============================================================
# Extract Python code and check syntax
# ============================================================

CODE_BLOCK_RE = re.compile(
    r"```(?:python)?[ \t]*\r?\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)


def extract_code(text: str) -> tuple[str | None, bool]:
    text = text.strip()

    blocks = CODE_BLOCK_RE.findall(text)
    code = blocks[-1].strip() if blocks else text

    # Remove common accidental leading labels.
    code = re.sub(r"^\s*Python\s*:\s*", "", code, flags=re.IGNORECASE)

    if not code:
        return None, False

    try:
        compile(code, "<generated>", "exec")
        return code, True
    except SyntaxError:
        return code, False


# ============================================================
# Save results
# ============================================================
def check_python_syntax(code: str) -> tuple[bool, str | None]:
    if not code.strip():
        return False, "Empty response"

    try:
        compile(
            code,
            "<model-output>",
            "exec",
            dont_inherit=True,
        )
    except SyntaxError as exc:
        return (
            False,
            f"{type(exc).__name__}: {exc.msg} "
            f"(line {exc.lineno})",
        )
    except ValueError as exc:
        return False, f"ValueError: {exc}"

    return True, None

def save_results(
    out_path: Path,
    model: str,
    fewshot: bool,
    results: list[dict[str, Any]],
) -> dict[str, Any]:

    bundle = {
        "meta": {
            "model": model,
            "mode": "fewshot" if fewshot else "zeroshot",
            "fewshot": fewshot,
            "n_shots": len(FEWSHOT_EXAMPLES) if fewshot else 0,
            "n_tasks": len(results),
            "endpoint": OLLAMA_CHAT_URL,
            "created_at": datetime.now(timezone.utc).isoformat(),
        },
        "results": results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return bundle


# ============================================================
# Run evaluation
# ============================================================

def run(
    tasks: list[dict[str, Any]],
    model: str,
    fewshot: bool,
    temperature: float,
    seed: int | None,
    out_path: Path,
) -> dict[str, Any]:

    results: list[dict[str, Any]] = []

    for index, task in enumerate(tasks, start=1):
        task_id = task["id"]
        instruction = task["instruction"]
        messages = build_messages(instruction, fewshot=fewshot)

        print(
            f"[{index}/{len(tasks)}] {task_id} ... ",
            end="",
            flush=True,
        )

        record: dict[str, Any] = {
            "task_id": task_id,
            "instruction": instruction,
            "fewshot": fewshot,
            "n_shots": len(FEWSHOT_EXAMPLES) if fewshot else 0,
        }

        try:
            output = call_ollama(
                messages=messages,
                model=model,
                temperature=temperature,
                seed=seed,
            )

            code, syntax_valid = extract_code(output["content"])

            record.update({
                "status": "ok",
                "raw_response": output["content"],
                "extracted_code": code,
                "syntax_valid": syntax_valid,
                "metrics": {
                    "total_seconds": output["total_seconds"],
                    "tokens_per_second": output["tokens_per_second"],
                    "eval_count": output["eval_count"],
                    "prompt_eval_count": output["prompt_eval_count"],
                },
            })

            print(
                f"done ({output['total_seconds']}s, "
                f"syntax_valid={syntax_valid})"
            )

        except Exception as exc:
            record.update({
                "status": "error",
                "error": f"{type(exc).__name__}: {exc}",
            })
            print(f"ERROR: {exc}")

        record["timestamp"] = datetime.now(timezone.utc).isoformat()
        results.append(record)

        # Save after each task so partial results are preserved.
        save_results(out_path, model, fewshot, results)

    return save_results(out_path, model, fewshot, results)


# ============================================================
# Load custom tasks
# ============================================================

def load_tasks(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"ไม่พบไฟล์ tasks: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if not isinstance(data, list):
        raise ValueError(
            "ไฟล์ tasks ต้องเป็น JSON array ของ object ที่มี id และ instruction"
        )

    for item in data:
        if not isinstance(item, dict):
            raise ValueError("แต่ละ task ต้องเป็น JSON object")
        if "id" not in item or "instruction" not in item:
            raise ValueError("ทุก task ต้องมี id และ instruction")

    return data


# ============================================================
# CLI
# ============================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Thai query to Python code using Ollama"
    )

    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"ชื่อโมเดล Ollama (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--tasks",
        type=Path,
        default=DEFAULT_TASKS_PATH,
        help=f"ไฟล์ JSON ที่เก็บคำสั่งภาษาไทย (default: {DEFAULT_TASKS_PATH.name})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="ไฟล์ JSON สำหรับบันทึกผลลัพธ์",
    )
    parser.add_argument(
        "--no-fewshot",
        action="store_true",
        help="ปิดตัวอย่าง Few-shot เพื่อทดสอบ Zero-shot",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    args = parser.parse_args()
    fewshot = not args.no_fewshot

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model)
    mode = "fewshot" if fewshot else "zeroshot"

    out_path = args.out or Path(
        f"results_{safe_model}_{mode}_{timestamp}.json"
    )

    try:
        tasks = load_tasks(args.tasks)
    except Exception as exc:
        print(f"อ่านไฟล์ tasks ไม่สำเร็จ: {exc}", file=sys.stderr)
        return 2

    print(f"Model: {args.model}")
    print(f"Mode: {mode}")
    print(f"Few-shot examples: {len(FEWSHOT_EXAMPLES) if fewshot else 0}")
    print(f"Tasks: {len(tasks)}")
    print(f"Output: {out_path}")

    try:
        requests.get(
            "http://localhost:11434/api/tags",
            timeout=5,
        ).raise_for_status()
    except Exception as exc:
        print(
            f"เชื่อมต่อ Ollama ไม่ได้: {exc}\n"
            "โปรดตรวจสอบว่า Ollama กำลังทำงานอยู่",
            file=sys.stderr,
        )
        return 2

    bundle = run(
        tasks=tasks,
        model=args.model,
        fewshot=fewshot,
        temperature=args.temperature,
        seed=args.seed,
        out_path=out_path,
    )

    results = bundle["results"]
    successful = sum(r.get("status") == "ok" for r in results)
    syntax_valid = sum(bool(r.get("syntax_valid")) for r in results)

    print(f"\nบันทึกผลลัพธ์แล้ว: {out_path}")
    print(f"เรียกโมเดลสำเร็จ: {successful}/{len(tasks)}")
    print(f"Python syntax ถูกต้อง: {syntax_valid}/{len(tasks)}")

    return 0 if successful == len(tasks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
