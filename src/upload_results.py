import json 
from datasets import load_dataset, Dataset
import os 
import sys

def load_and_upload(model_name):
    filepath = f"result_dirs/wild_bench/{model_name}.json"
    if not os.path.exists(filepath):
        print(f"File {filepath} does not exist.")
        return
    with open(filepath, "r") as f:
        data = json.load(f)
    dataset = Dataset.from_list(data)
    dataset.push_to_hub(
        repo_id="WildEval/WildBench-Results",
        config_name=model_name,
        split="train",
        token=os.environ.get("HUGGINGFACE_TOKEN"),
        commit_message=f"Add {model_name} results.",
    )
    print(f"Uploaded {model_name} results.")

load_and_upload(sys.argv[1])

"""
python src/upload_results.py tulu-2-dpo-70b
python src/upload_results.py gpt-3.5-turbo-0125
python src/upload_results.py gpt-4-0125-preview

python src/upload_results.py Mixtral-8x7B-Instruct-v0.1
python src/upload_results.py Mistral-7B-Instruct-v0.2
python src/upload_results.py Yi-34B-Chat
python src/upload_results.py vicuna-13b-v1.5
"""