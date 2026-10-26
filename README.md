# QLoRA Finetuning

Thai Query → Python Code generation experiments using Ollama.

## Files

- `fewshot_codegen.py` — Thai instructions → Python code via Ollama (few-shot / zero-shot prompting, syntax validation, JSON result output)
- `tasks_th.json` — dataset of Thai task instructions (`id` + `instruction`); swap datasets with `--tasks path/to/other.json`

## Usage

```bash
pip install requests
# Ollama must be running locally (http://localhost:11434)

python fewshot_codegen.py                        # few-shot (default)
python fewshot_codegen.py --no-fewshot           # zero-shot
python fewshot_codegen.py --model qwen2.5-coder:3b
python fewshot_codegen.py --tasks tasks_th.json --out results.json
```
