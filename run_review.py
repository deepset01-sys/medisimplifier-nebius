import sys
import anthropic
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

README = Path(__file__).parent / "README.md"
PROMPT = Path(__file__).parent / "review_prompt.txt"
TRAIN = Path(__file__).parent / "src/train.py"
EVALUATE = Path(__file__).parent / "src/evaluate.py"

readme_text = README.read_text(encoding="utf-8")
review_prompt = PROMPT.read_text(encoding="utf-8")
train_text = TRAIN.read_text(encoding="utf-8")
eval_text = EVALUATE.read_text(encoding="utf-8")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--model", default="claude-opus-4-7")
args = parser.parse_args()

client = anthropic.Anthropic()

with client.messages.stream(
    model=args.model,
    max_tokens=4000,
    messages=[
        {
            "role": "user",
            "content": (
                f"{review_prompt}\n\n---\n\nHere is the README.md to review:\n\n{readme_text}"
                f"\n\n---\n\nHere is src/train.py:\n\n{train_text}"
                f"\n\n---\n\nHere is src/evaluate.py:\n\n{eval_text}"
            ),
        }
    ],
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)

print()
