# Copyright 2026 Mark Alan Boykin
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

"""Independent Transformers fixtures; no download or model-supplied code.

Run with the pinned reference environment, passing the output directory.
Weights deliberately exercise nonzero biases and nonuniform QK norms.
"""
import hashlib
import json
from pathlib import Path
import sys
import time
import os

import torch
import transformers
from safetensors.torch import save_file
from transformers import (
    LlamaConfig, LlamaForCausalLM, MistralConfig, MistralForCausalLM,
    Qwen2Config, Qwen2ForCausalLM, Qwen3Config, Qwen3ForCausalLM,
    Qwen2MoeConfig, Qwen2MoeForCausalLM,
)


def main(destination):
    torch.set_num_threads(4)
    destination.mkdir(parents=True, exist_ok=True)
    cases = [
        ("llama", LlamaConfig, LlamaForCausalLM, {}),
        ("llama3", LlamaConfig, LlamaForCausalLM, {"rope_scaling": {
            "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0, "original_max_position_embeddings": 128}}),
        ("linear", LlamaConfig, LlamaForCausalLM,
         {"rope_scaling": {"rope_type": "linear", "factor": 4.0}}),
        ("mistral", MistralConfig, MistralForCausalLM,
         {"head_dim": 12, "sliding_window": 3}),
        ("qwen2", Qwen2Config, Qwen2ForCausalLM,
         {"use_sliding_window": True, "sliding_window": 3, "max_window_layers": 1}),
        ("qwen3", Qwen3Config, Qwen3ForCausalLM,
         {"head_dim": 12, "layer_types": ["sliding_attention", "full_attention"],
          "sliding_window": 3, "use_sliding_window": True}),
        ("qwen2_moe", Qwen2MoeConfig, Qwen2MoeForCausalLM,
         {"num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":12,
          "shared_expert_intermediate_size":20,"norm_topk_prob":False,"decoder_sparse_step":1}),
        ("qwen2_moe_normalized", Qwen2MoeConfig, Qwen2MoeForCausalLM,
         {"num_experts":4,"num_experts_per_tok":2,"moe_intermediate_size":12,
          "shared_expert_intermediate_size":20,"norm_topk_prob":True,"decoder_sparse_step":2}),
    ]
    references = []
    for name, configuration, implementation, extra in cases:
        torch.manual_seed(8301)
        config = configuration(vocab_size=32, hidden_size=16, intermediate_size=24,
                               num_hidden_layers=2, num_attention_heads=2,
                               num_key_value_heads=1, max_position_embeddings=256,
                               tie_word_embeddings=False, rms_norm_eps=1e-5,
                               rope_theta=10000.0, eos_token_id=None,
                               attention_dropout=0.0, **extra)
        config._attn_implementation = "eager"
        model = implementation(config).eval()
        with torch.no_grad():
            for key, parameter in model.named_parameters():
                if key.endswith(".bias"):
                    parameter.copy_(torch.linspace(-0.15, 0.12, parameter.numel()).reshape(parameter.shape))
                if key.endswith(("q_norm.weight", "k_norm.weight")):
                    parameter.copy_(torch.linspace(0.7, 1.3, parameter.numel()))
        folder = destination / name
        folder.mkdir(exist_ok=True)
        config.save_pretrained(folder)
        weights = {key: tensor.detach().contiguous().clone()
                   for key, tensor in model.state_dict().items()}
        save_file(weights, folder / "model.safetensors")
        ids = ([1, 4, 9, 2, 7, 10, 3, 11, 5] * 15)[:130] if name == "llama3" else [1, 4, 9, 2, 7, 10, 3, 11, 5]
        sequence = list(ids)
        generated = []
        with torch.no_grad():
            initial = model(torch.tensor([sequence]), use_cache=False).logits[0, -1].tolist()
            for _ in range(4):
                next_id = int(model(torch.tensor([sequence]), use_cache=False).logits[0, -1].argmax())
                generated.append(next_id)
                sequence.append(next_id)
        references.append({"name": name, "input_ids": ids, "last_logits": initial,
                           "generated_ids": generated,
                           "weights_sha256": hashlib.sha256((folder / "model.safetensors").read_bytes()).hexdigest()})
    (destination / "reference.json").write_text(json.dumps({
        "torch": torch.__version__, "transformers": transformers.__version__,
        "implementation": "Transformers eager, float32, full recomputation per generated token",
        "cases": references,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(references), "bytes": sum(p.stat().st_size for p in destination.rglob('*') if p.is_file())}))


def real(directory, template_path, model_id, revision, output):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    assert directory.name == revision
    assert directory.parent.parent.name == 'models--' + model_id.replace('/', '--')
    torch.set_num_threads(4)
    template = json.loads(template_path.read_text(encoding="utf-8"))
    tokenizer = AutoTokenizer.from_pretrained(directory, local_files_only=True, trust_remote_code=False)
    dtype=torch.bfloat16 if os.environ.get('ESP_REFERENCE_DTYPE')=='bf16' else torch.float32
    offload=os.environ.get('ESP_REFERENCE_OFFLOAD_DIR')
    placement={}
    if offload:
        cache=Path(offload);cache.mkdir(parents=True,exist_ok=True)
        (cache/'CACHEDIR.TAG').write_text('Signature: 8a477f597d28d172789f06886806bc55\n# Regenerable Accelerate offload cache, not a retained receipt.\n',encoding='utf-8')
        placement={'device_map':'auto','max_memory':{'cpu':int(os.environ.get('ESP_REFERENCE_CPU_BYTES','12000000000'))},
            'offload_folder':str(cache),'offload_state_dict':True}
    model = AutoModelForCausalLM.from_pretrained(directory, local_files_only=True,
        trust_remote_code=False, torch_dtype=dtype, attn_implementation="eager",**placement).eval()
    print(json.dumps({'loaded':model_id,'dtype':str(dtype),'device_map':getattr(model,'hf_device_map',None)}),flush=True)
    eos = model.config.eos_token_id
    stops = set(eos if isinstance(eos, list) else [eos])
    declared_eos=tokenizer.get_vocab().get(template['eos'])
    if declared_eos is not None: stops.add(declared_eos)
    rows = []
    for question in ["What is the capital of France?", "What is two plus two?"]:
        messages = []
        prompt = template['bos']
        if template['system_prompt']:
            messages.append({'role':'system', 'content':template['system_prompt']})
            prompt += template['system']['prefix'] + template['system_prompt'] + template['system']['suffix']
        messages.append({'role':'user', 'content':question})
        prompt += template['user']['prefix'] + question + template['user']['suffix'] + template['assistant']['prefix']
        official = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        assert prompt == official, 'data template differs from pinned tokenizer template'
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        sequence = list(ids)
        tokens = []
        start = time.monotonic()
        print(json.dumps({'question':question,'prompt_tokens':len(ids),'started':True}),flush=True)
        past=None
        current=torch.tensor([sequence])
        with torch.no_grad():
            for _ in range(16):
                result=model(current if offload else torch.tensor([sequence]),past_key_values=past,use_cache=bool(offload))
                next_id = int(result.logits[0,-1].argmax())
                print(json.dumps({'question':question,'step':len(tokens)+1,'token_id':next_id,'seconds':time.monotonic()-start}),flush=True)
                if next_id in stops:
                    break
                tokens.append(next_id)
                sequence.append(next_id)
                if offload:
                    past=result.past_key_values
                    current=torch.tensor([[next_id]])
        rows.append({'question':question, 'prompt':prompt, 'prompt_ids':ids, 'token_ids':tokens,
            'text':tokenizer.decode(tokens, skip_special_tokens=True), 'seconds':time.monotonic()-start})
    artifacts = {}
    filenames=['config.json','generation_config.json','tokenizer.json','tokenizer_config.json']
    filenames += ['model.safetensors'] if (directory/'model.safetensors').exists() else ['model.safetensors.index.json']+[p.name for p in sorted(directory.glob('model-*.safetensors'))]
    for name in filenames:
        with (directory/name).open('rb') as source:
            artifacts[name] = hashlib.file_digest(source,'sha256').hexdigest()
        print(json.dumps({'hashed':name}),flush=True)
    output.write_text(json.dumps({'model_id':model_id,'revision':revision,
        'torch':torch.__version__,'transformers':transformers.__version__,
        'dtype':str(dtype),'attention':'eager','max_tokens':16,
        'reference':'cached Transformers greedy tokens' if offload else 'full recomputation per greedy token', 'official_template_match':True,
        'template_sha256':hashlib.sha256(template_path.read_bytes()).hexdigest(),
        'artifacts':artifacts,'rows':rows},indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'model':model_id, 'rows':len(rows),'output':str(output)}),flush=True)


if __name__ == "__main__":
    if sys.argv[1] == 'real':
        real(Path(sys.argv[2]),Path(sys.argv[3]),sys.argv[4],sys.argv[5],Path(sys.argv[6]))
    else:
        main(Path(sys.argv[1]))
