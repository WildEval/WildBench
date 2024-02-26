import json 
from datasets import load_dataset, Dataset
import os 
import sys

def load_and_upload(model_name, evaluator="gpt-4-0125-preview", reference="gpt-3.5-turbo-0125"):
    filepath = f"evaluation/results/eval={evaluator}/ref={reference}/{model_name}.json"
    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist.")
        return
    with open(filepath, "r") as f:
        data = json.load(f)
    print(f"Loaded {model_name} results with {len(data)} samples.")
    if len(data) != 1024:
        print(f"Expected 1024 samples, got {len(data)}. Exit!")
        return
    dataset = Dataset.from_list(data)
    dataset.push_to_hub(
        repo_id="WildEval/WildBench-Evaluation",
        config_name=model_name+"-eval="+evaluator+"-ref="+reference,
        split=f"train",
        token=os.environ.get("HUGGINGFACE_TOKEN"),
        commit_message=f"Add {model_name} results. Evaluated with {evaluator} and referenced with {reference}.",
    )
    print(f"Uploaded {model_name} results.")

load_and_upload(sys.argv[3], sys.argv[1], sys.argv[2])

"""
python src/upload_evaluation.py gpt-4-0125-preview gpt-3.5-turbo-0125 gpt-4-0125-preview
python src/upload_evaluation.py gpt-4-0125-preview gpt-3.5-turbo-0125 tulu-2-dpo-70b
python src/upload_evaluation.py gpt-4-0125-preview gpt-3.5-turbo-0125 Mistral-7B-Instruct-v0.2
python src/upload_evaluation.py gpt-4-0125-preview gpt-3.5-turbo-0125 Mixtral-8x7B-Instruct-v0.1
python src/upload_evaluation.py gpt-4-0125-preview gpt-3.5-turbo-0125 Yi-34B-Chat
python src/upload_evaluation.py gpt-4-0125-preview gpt-3.5-turbo-0125 vicuna-13b-v1.5

python src/upload_evaluation.py gpt-4-0125-preview gpt-3.5-turbo-0125 Llama-2-70b-chat-hf
# python src/upload_evaluation.py gpt-4-0125-preview gpt-3.5-turbo-0125 Llama-2-13b-chat-hf
python src/upload_evaluation.py gpt-4-0125-preview gpt-3.5-turbo-0125 Llama-2-7b-chat-hf

python src/upload_evaluation.py gpt-4-0125-preview gpt-3.5-turbo-0125 Mistral-7B-Instruct-v0.1
python src/upload_evaluation.py gpt-4-0125-preview gpt-3.5-turbo-0125 gemma-7b-it
python src/upload_evaluation.py gpt-4-0125-preview gpt-3.5-turbo-0125 gemma-2b-it
"""