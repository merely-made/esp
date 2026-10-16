# Copyright 2026 Mark Alan Boykin
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Offline I6 preparation and independent float32 Transformers reference.

reference_layer_streaming.py prepare SNAPSHOT CONVERTED_DIR TEMPLATE_JSON
reference_layer_streaming.py reference CONVERTED_DIR TEMPLATE_JSON OUTPUT_JSON

The snapshot must be openlm-research/open_llama_3b_v2 at the revision below.
Preparation uses torch's restricted weights-only reader and memory mapping,
then writes ordinary safetensors shards without changing tensor values.
Neither command downloads files or executes model-supplied Python.
"""

import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

import torch
import transformers
import psutil
from transformers import AutoModelForCausalLM, AutoTokenizer
from safetensors.torch import save_file

MODEL = "openlm-research/open_llama_3b_v2"
REVISION = "bce5d60d3b0c68318862270ec4e794d83308d80a"
UPSTREAM_WEIGHT_SHA256 = "32268cfa47263b8abef86c9238e3ec7d64faabde504413b8e0a4d56002d76ec0"
QUESTIONS = ["What is the capital of France?", "What is two plus two?"]


def digest(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def prompt(template, question):
    assert not template["system_prompt"]
    return (template["bos"] + template["user"]["prefix"] + question
            + template["user"]["suffix"] + template["assistant"]["prefix"])


def prepare(source, destination, template_path):
    assert source.name == REVISION, "use the exact upstream snapshot directory"
    destination.mkdir(parents=True, exist_ok=True)
    binary_hash = digest(source/"pytorch_model.bin")
    assert binary_hash == UPSTREAM_WEIGHT_SHA256, "exact upstream LFS weight identity"
    template = json.loads(template_path.read_text(encoding="utf-8"))
    for name in ["config.json", "generation_config.json"]:
        shutil.copyfile(source/name, destination/name)
    slow = AutoTokenizer.from_pretrained(source, local_files_only=True,
                                        trust_remote_code=False, use_fast=False, legacy=True)
    fast = AutoTokenizer.from_pretrained(source, local_files_only=True,
                                        trust_remote_code=False, use_fast=True, legacy=True)
    # OpenLlama documents legacy tokenizer differences. Do not compare two
    # model executions with different input IDs for these receipt prompts.
    for question in QUESTIONS:
        text = prompt(template, question)
        assert fast.encode(text, add_special_tokens=False) == slow.encode(text, add_special_tokens=False)
    fast.save_pretrained(destination)
    state = torch.load(source/"pytorch_model.bin", map_location="cpu", weights_only=True, mmap=True)
    groups, group, size = [], {}, 0
    for name, tensor in state.items():
        nbytes = tensor.numel() * tensor.element_size()
        if group and size + nbytes > 1024**3:
            groups.append(group)
            group, size = {}, 0
        group[name] = tensor
        size += nbytes
    if group:
        groups.append(group)
    weight_map = {}
    total = sum(t.numel() * t.element_size() for t in state.values())
    for index, group in enumerate(groups, 1):
        name = f"model-{index:05d}-of-{len(groups):05d}.safetensors"
        save_file(group, destination/name, metadata={"format": "pt"})
        weight_map.update({key: name for key in group})
    (destination/"model.safetensors.index.json").write_text(json.dumps(
        {"metadata": {"total_size": total}, "weight_map": weight_map}, indent=2)+"\n", encoding="utf-8")
    upstream = {name: digest(source/name) for name in ["config.json", "tokenizer.model"]}
    upstream["pytorch_model.bin"] = binary_hash
    (destination/"conversion.json").write_text(json.dumps({
        "model_id": MODEL, "revision": REVISION, "upstream_sha256": upstream,
        "weight_bytes": total, "shards": len(groups), "dtype_preserved": True,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "fixed_prompt_fast_slow_tokenizer_parity": True,
    }, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"prepared": str(destination), "shards": len(groups), "weight_bytes": total}), flush=True)


def reference(directory, template_path, output):
    conversion = json.loads((directory/"conversion.json").read_text(encoding="utf-8"))
    assert conversion["revision"] == REVISION
    template = json.loads(template_path.read_text(encoding="utf-8"))
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(directory, local_files_only=True,
        trust_remote_code=False, torch_dtype=torch.float32, low_cpu_mem_usage=True,
        attn_implementation="eager").eval()
    weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    rows = []
    for question in QUESTIONS:
        text = prompt(template, question)
        inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt")
        start = time.monotonic()
        with torch.inference_mode():
            result = model.generate(**inputs, max_new_tokens=8, do_sample=False)
        tokens = result[0, inputs.input_ids.shape[1]:].tolist()
        if tokens and tokens[-1] == model.config.eos_token_id:
            tokens.pop()
        rows.append(dict(question=question, prompt=text, prompt_ids=inputs.input_ids[0].tolist(),
            token_ids=tokens, text=tokenizer.decode(tokens, skip_special_tokens=True),
            seconds=time.monotonic()-start))
    names = ["config.json", "tokenizer.json", "tokenizer_config.json", "model.safetensors.index.json"]
    names += sorted(set(json.loads((directory/"model.safetensors.index.json").read_text())["weight_map"].values()))
    artifacts = {name: digest(directory/name) for name in names}
    receipt = dict(model_id=MODEL, revision=REVISION, torch=torch.__version__,
        transformers=transformers.__version__, dtype="float32", attention="eager",
        max_tokens=8, temperature=0, artifacts=artifacts, rows=rows,
        resident_f32_weight_bytes=weight_bytes, conversion=conversion,
        process_peak_working_set_bytes=getattr(psutil.Process().memory_info(), "peak_wset", None),
        host_memory_bytes=psutil.virtual_memory().total,
        template_sha256=digest(template_path))
    output.write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "resident_f32_weight_bytes": weight_bytes,
                      "rows": len(rows), "seconds": [row["seconds"] for row in rows]}), flush=True)


if __name__ == "__main__":
    mode, *arguments = sys.argv[1:]
    if mode == "prepare" and len(arguments) == 3:
        prepare(*map(Path, arguments))
    elif mode == "reference" and len(arguments) == 3:
        reference(*map(Path, arguments))
    else:
        raise SystemExit(__doc__)
