#!/usr/bin/env python3
"""
Generate simple JSONL benchmark data for guidellm.

Creates datasets with just prompt and output_tokens_count columns.
"""

import json
from pathlib import Path

# Configuration
NUM_REQUESTS = 100
OUTPUT_TOKENS = 500  # Reduced to fit in 2048 context
TARGET_PROMPT_TOKENS = 500  # Reduced to fit in 2048 context

# Simple prompts that will result in approximately 1000 tokens
# Based on 2x character-to-token ratio observed earlier
CHARS_PER_TOKEN = 2
target_chars = TARGET_PROMPT_TOKENS * CHARS_PER_TOKEN

# Generate a prompt template
base_prompt = "Count from 1 to 100: "
# Fill the rest with numbers to reach target length
filler = " ".join(str(i) for i in range(1, 10000))
prompt_template = base_prompt + filler[: target_chars - len(base_prompt)]

print(f"Generating {NUM_REQUESTS} requests...")
tokens = len(prompt_template) // CHARS_PER_TOKEN
print(f"Prompt length: {len(prompt_template)} characters (~{tokens} tokens)")
print(f"Output tokens: {OUTPUT_TOKENS}")

# Create JSONL data
data = []
for i in range(NUM_REQUESTS):
    # Vary the prompt slightly to avoid caching
    prompt = f"Request {i}: {prompt_template}"

    data.append(
        {
            "prompt": prompt,
            "output_tokens_count": OUTPUT_TOKENS,
        }
    )

# Write to file (same data used for both freeform and structured)
output_path = Path("/tmp/structured-output-data.jsonl")
with output_path.open("w") as f:
    for item in data:
        f.write(json.dumps(item) + "\n")

print(f"\nData saved to {output_path}")
print(f"Total requests: {len(data)}")

# Create symlinks for clarity
Path("/tmp/structured-output-data-freeform.jsonl").unlink(missing_ok=True)
Path("/tmp/structured-output-data-structured.jsonl").unlink(missing_ok=True)
Path("/tmp/structured-output-data-freeform.jsonl").symlink_to(output_path)
Path("/tmp/structured-output-data-structured.jsonl").symlink_to(output_path)

print("Created symlinks:")
print("  /tmp/structured-output-data-freeform.jsonl -> structured-output-data.jsonl")
print("  /tmp/structured-output-data-structured.jsonl -> structured-output-data.jsonl")
