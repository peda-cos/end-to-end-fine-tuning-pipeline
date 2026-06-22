from __future__ import annotations

import yaml

from ftpipe.config import load_config
from ftpipe.emit.axolotl import emit_axolotl


def test_axolotl_config_has_real_stack_flags(tmp_path):
    cfg = load_config("configs/domainx.yaml")
    paths = emit_axolotl(cfg, tmp_path)
    payload = yaml.safe_load(paths["sft"].read_text(encoding="utf-8"))
    assert payload["base_model"] == "meta-llama/Llama-3.1-8B-Instruct"
    assert payload["sample_packing"] is True
    assert payload["attn_implementation"] == "flash_attention_3"
    assert payload["deepspeed"].endswith("zero3.json")
    assert payload["num_epochs"] == 3
    assert payload["datasets"][0]["path"] == "outputs/data/sft_train.jsonl"
    lora = yaml.safe_load(paths["sft_lora"].read_text(encoding="utf-8"))
    assert lora["adapter"] == "lora"
    assert lora["lora_target_linear"] is True
