# Copyright 2026 Mark Alan Boykin
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Independent I2 receipt. Run in a CPU torch/transformers environment.

python reference_smollm2.py MODEL_SNAPSHOT OUTPUT_JSON
Uses local artifacts only; never downloads or executes model-supplied code.
"""
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

model_dir, output = map(Path, sys.argv[1:])
artifacts = {name: hashlib.file_digest((model_dir/name).open("rb"), "sha256").hexdigest()
             for name in ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors"]}
assert artifacts == {
    "config.json": "224f72354f10d617a359cc82ad15a3c96e866b9b2ffadb81997eeea9e88e22ee",
    "tokenizer.json": "9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c",
    "tokenizer_config.json": "4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df",
    "model.safetensors": "e6bffe7435d7ddc10fd3b9a9efd429dafbacb1cb17015fb5562664e7532bf86e",
}, "the input files must match the pinned SmolLM2-360M-Instruct snapshot"
torch.set_num_threads(4)
tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(
    model_dir, local_files_only=True, torch_dtype=torch.float32,
    attn_implementation="eager",
).eval()
rows = []
for question in ["What is the capital of France?", "Count from one to five.", "What color is a clear daytime sky?"]:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], tokenize=False, add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
    start = time.monotonic()
    with torch.inference_mode():
        result = model.generate(**inputs, max_new_tokens=16, do_sample=False)
    tokens = result[0, inputs.input_ids.shape[1]:].tolist()
    if tokens and tokens[-1] == model.config.eos_token_id:
        tokens.pop()
    rows.append(dict(question=question, prompt=prompt, prompt_ids=inputs.input_ids[0].tolist(),
                     token_ids=tokens, text=tokenizer.decode(tokens, skip_special_tokens=True),
                     seconds=time.monotonic()-start))
receipt = dict(model_id="HuggingFaceTB/SmolLM2-360M-Instruct",
               revision="a10cc1512eabd3dde888204e902eca88bddb4951",
               torch=torch.__version__, transformers=transformers.__version__,
               dtype="float32", attention="eager", max_tokens=16, temperature=0,
               artifacts=artifacts,
               rows=rows)
output.write_text(json.dumps(receipt, indent=2)+"\n")
print(json.dumps({"output": str(output), "rows": len(rows), "seconds": [r["seconds"] for r in rows]}))
