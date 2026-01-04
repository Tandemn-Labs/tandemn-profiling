# TENSOR_DEGREE=[1,4,8]
# PIPELINE_DEGREE=[1,2,3,4]
# MAX_INPUT_LENGTH=[4096,10000,30000] # 4096 IS SMALL, MEDIUM IS 10000, LARGE IS 32000, Do we need 128k Input Length?
# MAX_OUTPUT_LENGTH=[1024,4096,7000] # 1024 IS SMALL, MEDIUM IS 2048, LARGE IS 8192, HUGE IS 32768, Do we need 128k Output Length?
TENSOR_DEGREE=[1,2,4,8]
PIPELINE_DEGREE=[1,2,3,4]
MAX_INPUT_LENGTH=[8192] # 4096 IS SMALL, MEDIUM IS 10000, LARGE IS 32000, Do we need 128k Input Length?
MAX_OUTPUT_LENGTH=[2048] # 1024 IS SMALL, MEDIUM IS 2048, LARGE IS 8192, HUGE IS 32768, Do we need 128k Output Length?

BASE_MODELS=['deepseek-ai/DeepSeek-R1-Distill-Llama-70B']
# FOUR_BIT_MODELS=['unsloth/gpt-oss-120b','unsloth/QwQ-32B-unsloth-bnb-4bit', 'unsloth/gpt-oss-120b-unsloth-bnb-4bit']

EXPERIMENT_CSV_FILENAME='experiment_l40_llama.csv'

with open(EXPERIMENT_CSV_FILENAME, 'w') as f:
    f.write('tensor_degree,pipeline_degree,max_input_length,max_output_length,model\n')

    for tensor_degree in TENSOR_DEGREE:
        for pipeline_degree in PIPELINE_DEGREE:
            for max_input_length in MAX_INPUT_LENGTH:
                for max_output_length in MAX_OUTPUT_LENGTH:
                    for base_model in BASE_MODELS:
                        if tensor_degree == 1 and pipeline_degree == 1:
                            print(f'Skipping tp1-pp1 configuration. {base_model} will not fit in a single GPU.')
                            continue
                        # if tensor_degree == 8 and pipeline_degree == 4:
                        #     continue
                        f.write(f'{tensor_degree},{pipeline_degree},{max_input_length},{max_output_length},{base_model}\n')


print(f"Experiment CSV generated: {EXPERIMENT_CSV_FILENAME}")