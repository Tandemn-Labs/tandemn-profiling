import os 
from pydoc import tempfilepager
import time
import json
from vllm import LLM, SamplingParams
import argparse
from datasets import load_dataset
from vllm.config import KVTransferConfig
import tempfile

os.environ["VLLM_USE_V1"] = "1" 

def read_prompts(num_prompts):
    billsum = load_dataset("billsum", split="ca_test")
    billsum = billsum.train_test_split(test_size=0.2, seed=42)
    prompts = []
    for i in range(num_prompts):
        prompts.append("Summarize This: " + billsum["train"][i]["text"][:1000]) # hardcoded input length
    return prompts

def setup_lmcache_s3(model, dtype, s3_bucket, s3_region):
    yaml_config = f"""chunk_size: 256
local_cpu: False
save_unfull_chunk: False
remote_url: "s3://{s3_bucket}"
remote_serde: "naive"
blocking_timeout_secs: 10

extra_config:
  s3_region: "{s3_region}"
  s3_num_io_threads: 64
  save_chunk_meta: False
"""
    
    config_file = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
    config_file.write(yaml_config)
    config_file.close()
    
    os.environ["LMCACHE_CONFIG_FILE"] = config_file.name
    os.environ["LMCACHE_USE_EXPERIMENTAL"] = "True"
    os.environ["PYTHONHASHSEED"] = "0"  # CRITICAL!
    
    print("LMCACHE + S3 SETUP COMPLETE")
    return config_file.name


def apply_chat_template(llm, prompts):
    """Apply chat template to prompts"""
    tokenizer = llm.get_tokenizer()
    
    formatted_prompts = []
    for prompt in prompts:
        messages = [
            {"role": "user", "content": prompt}
        ]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_prompts.append(formatted)
    
    return formatted_prompts

def do_prefill(model, dtype, prompts, output_file="prefill_prompts.json"):
    print("DOING PREFILL")
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=1, # force 1 for prefill
        ignore_eos=True # only for testing
    )
    # create the llm with kv_transfer_config
    kv_transfer_config = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )
    llm = LLM(
        model=model,
        dtype=dtype,
        kv_transfer_config=kv_transfer_config,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        max_model_len=8000, # we can keep on changing this
        enable_prefix_caching=True,
        enforce_eager=True # keep the max_batched_tokens and max_num_seqs as the default
    )
    prompts = apply_chat_template(llm, prompts)
    outputs = llm.generate(prompts, sampling_params)
    new_prompts = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        new_prompts.append(prompt + generated_text)
    with open(output_file, 'w') as f:
            json.dump({"prompts": new_prompts}, f, indent=2)
    
    return True

def do_decode(model, dtype, prompts, output_file="decode_prompts.json"):
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=2000,
        min_tokens=2000,
        ignore_eos=True
    )
    # create the llm with kv_transfer_config as kv_consumer
    kv_transfer_config = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )
    llm = LLM(
        model=model,
        dtype=dtype,
        kv_transfer_config=kv_transfer_config,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        max_model_len=8000, # we can keep on changing this
        enable_prefix_caching=True,
        enforce_eager=True # keep the max_batched_tokens and max_num_seqs as the default
    )
    # apply chat template
    # prompts = apply_chat_template(llm, prompts)
    outputs = llm.generate(prompts, sampling_params)
    # results = []
    # for output in outputs:
    #     prompt = output.prompt
    #     generated_text = output.outputs[0].text
    #     results.append(prompt + generated_text)
    # with open(output_file, 'w') as f:
    #     json.dump({"prompts": results}, f, indent=2)
    return True

def main():
    parser = argparse.ArgumentParser(description="Mooncake + LMCache with P/D Over Time")
    parser.add_argument("--mode", type=str, choices=['prefill', 'decode'], required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--s3-bucket", type=str, required=True, help="S3 bucket name")
    parser.add_argument("--s3-region", type=str, default="us-east-1", help="S3 region")
    parser.add_argument("--num_prompts", type=int, required=True, help="How much prompts do you need")
    parser.add_argument("--output-file", type=str, required=True, default="prefill_prompts.json", help="Output file")

    args = parser.parse_args()
    config_file = setup_lmcache_s3(args.model, args.dtype, args.s3_bucket, args.s3_region)    
    if args.mode == "prefill":
        prompts = read_prompts(args.num_prompts)
        do_prefill(args.model, args.dtype, prompts, args.output_file)

        # copy the json and send it to the second machine
        # scp_command = f"scp -i tandemn.pem {args.output_file} ubuntu@34.207.146.140:/home/ubuntu/hetarth"
        # os.system(scp_command)

    elif args.mode == "decode":
        with open(args.output_file, 'r') as f:
            data = json.load(f)
            prompts = data['prompts']
        # prompts = read_prompts(args.num_prompts)
        do_decode(args.model, args.dtype, prompts, args.output_file)
    
    os.unlink(config_file)

if __name__ == "__main__":
    main()


# Example (prefill):

# PYTHONHASHSEED=0 python s3_lmcache.py \
#   --mode prefill \
#   --model "Qwen/Qwen2.5-0.5B-Instruct" \
#   --dtype bfloat16 \
#   --s3-bucket tandemn-kv-data \
#   --s3-region us-east-1 \
#   --num_prompts 50 \
#   --output-file prefill_prompts.json


# Example (decode):

# PYTHONHASHSEED=0 python s3_lmcache.py \
#   --mode decode \
#   --model "Qwen/Qwen2.5-0.5B-Instruct" \
#   --dtype bfloat16 \
#   --s3-bucket tandemn-kv-data \
#   --s3-region us-east-1 \
#   --num_prompts 50 \
#   --output-file decode_prompts.json

