"""
This project presents our efforts towards interpretable mental health analysis with large language models (LLMs). In early works we comprehensively evaluate the zero-shot/few-shot performances of the latest LLMs such as ChatGPT and GPT-4 on generating explanations for mental health analysis. Based on the findings, we build the Interpretable Mental Health Instruction (IMHI) dataset with 105K instruction samples, the first multi-task and multi-source instruction-tuning dataset for interpretable mental health analysis on social media. Based on the IMHI dataset, We propose MentaLLaMA, the first open-source instruction-following LLMs for interpretable mental health analysis. MentaLLaMA can perform mental health analysis on social media data and generate high-quality explanations for its predictions. We also introduce the first holistic evaluation benchmark for interpretable mental health analysis with 19K test samples, which covers 8 tasks and 10 test sets. Our contributions are presented in these 2 papers:

Transformer Model Cache: os.env{TRANSFORMERS_CACHE}
Transformer Dataset:     os.env{HF_DATASETS_CACHE}
"""




from transformers import AutoTokenizer, GenerationConfig, pipeline, LlamaTokenizer, LlamaForCausalLM
from langchain.prompts import PromptTemplate
from datasets import Dataset
import pandas as pd
from transformers import logging as hf_logging
import torch
# hf_logging.set_verbosity_error()
import warnings
warnings.simplefilter("ignore")

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    llama_pipeline = pipeline('text-generation', model=MODEL_PATH, torch_dtype=torch.float16, device_map='auto')
    return tokenizer, llama_pipeline

def get_llm_response(example, **kwargs):
    prompt = "Consider this post: \"{context}\" Question: Does the poster suffer from depression?"
    sequences = llama_pipeline( [prompt.format(context=x) for x in example['text']], **kwargs)
    return dict(response=[x[0]['generated_text'] for x in sequences])



# tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH)
# model = LlamaForCausalLM.from_pretrained(MODEL_PATH, device_map='auto')
# model_config = GenerationConfig.from_pretrained(MODEL_PATH)

# prompt = "Hey, are you conscious? Can you talk to me?"
# inputs = tokenizer(prompt, return_tensors="pt")
# generate_ids = model.generate(inputs.input_ids, max_length=30, do_sample=False)
# print(tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
# print(1)

def generate_prompt(prompt, sys_prompt):
    return f"[INST] <<SYS>> {sys_prompt} <</SYS>> {prompt} [/INST]"
def get_sys_prompt():
    return "Answer the following question and output your choice in just one alphabetical letter. Do not give reasoning or put quotation marks or escape character \ in the output fields"
def get_choice_prompt(choice):
    return '\n'.join([f'{k}) {v}' for k,v in choice.items()])
symptom_choices = {
    'A': 'Persistent depressed mood', 
    'B': 'Loss of interest/pleasure', 
    'C': 'Significant weight change or appetite',
    'D': 'Insomnia or hypersomnia',
    'E': 'Fatigue or loss of energy',
    'F': 'Feelings of worthlessness or guilt',
    'G': 'Reduced concentration or indecisiveness',
    'H': 'Suicidal thoughts or attempts',
    'I': 'None of above'
}
def get_first_prompt(example):
    question = "Does the post suffer from any of the nine symptoms of depression, or 'None' if it does not show any depression symptoms. You should pick the most pertinent choice from the following"
    template = generate_prompt(
    """
    Context: {context}

    Question : {question}

    Choice: \n{choice}
    """
    , get_sys_prompt())
    output_format = {'choice': f'your suggested choice'}
    prompt = PromptTemplate(template=template, input_variables=['question', 'choice', 'context'])
    return dict(prompt=prompt.format(question=question, choice=get_choice_prompt(symptom_choices), context=example['text']))

def plot_sequence_lengths(data, max_length=1024):
    # remove post that is too long
    keep_indices = []
    for i, example in enumerate(data):
        if len(example['text']) < max_length:
            keep_indices.append(i)
    return keep_indices


@torch.no_grad()
def generate_response(model, tokenizer, test_data, device, batch_size, start_index):
    from tqdm import trange
    prompt = "Consider this post: \"{post}\" Question: Does this poster suffer from depression"
    n = len(test_data)
    for i in trange(start_index, n, batch_size):
        batch_data = test_data[i: min(i+batch_size, n)]
        inputs = tokenizer(prompt.format(post=batch_data['text']), return_tensors="pt", padding=True)
        input_ids = inputs.input_ids.to(device)
        attention_mask = inputs.attention_mask.to(device)
        generate_ids = model.generate(input_ids, attention_mask=attention_mask, max_length=2048)
        for j in range(generate_ids.shape[0]):
            truc_ids = generate_ids[j][len(input_ids[j]) :]
            response = tokenizer.decode(truc_ids, skip_special_tokens=True, spaces_between_special_tokens=False)
            yield response

if __name__ == '__main__':
    import argparse, joblib, deepspeed
    parser = argparse.ArgumentParser(description='llama')
    parser.add_argument( '-i', '--index', default=0, type=int)
    args, _ = parser.parse_known_args()
    batch_size = 16
    df_path = '/mnt/Data/data/SocialMedia/df_Depression.pkl'
    df = pd.read_pickle(df_path).iloc[:100]
    data = Dataset.from_pandas(df, preserve_index=False)
    data = data.select(plot_sequence_lengths(data))
    MODEL_PATH = 'klyang/MentaLLaMA-chat-7B'
    llama_config = GenerationConfig.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, padding_side='left')
    model = LlamaForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float64, device_map='cpu')
    engine = deepspeed.init_inference( model, mp_size = 1, dtype = torch.int8, replace_with_kernel_inject = True)
    model = engine.module
    tokenizer.pad_token = tokenizer.unk_token
    device = next(model.parameters()).device
    reasonings = []
    start_index = args.index
    if start_index != 0:
        reasonings = joblib.load('/mnt/Data/data/SocialMedia/llama_Depression.job')
    for reason in generate_response(model, tokenizer, data, device, batch_size, start_index):
        reasonings.append(reason)
        start_index += 1
        if start_index % 5000 == 1:
            joblib.dump(reasonings, '/mnt/Data/data/SocialMedia/llama_Depression.job')
    data = data.add_column('response', reasonings)
    data.to_pandas().to_pickle(df_path)

