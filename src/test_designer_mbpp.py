import argparse
import os
import json
from tqdm import tqdm
import copy
import anthropic
from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import time
from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

# Setting API parameters
client = anthropic.Anthropic(
    api_key=os.getenv("ANTHROPIC_API_KEY")
)

dataset = load_dataset("evalplus/mbppplus",split="test")
dataset = [entry for entry in dataset]
task_0_tests = "\n".join(dataset[0]["test_list"])
task_1_tests = "\n".join(dataset[1]["test_list"])

prompt_path = "./prompts/test_designer_mbpp_prompt_update.txt"
with open(prompt_path, "r") as f:
    construct_few_shot_prompt = f.read()

def preprocess_data(test_case_string):
    if f"```python" in test_case_string:
        test_case_string = test_case_string[test_case_string.find(f"```python")+len(f"```python"):]
        test_case_string = test_case_string[:test_case_string.find("```")]
    return test_case_string

# Function to fetch completion
def fetch_completion(data_entry, model, lg,times=5):
    global construct_few_shot_prompt
    if "need_reproduce" in data_entry.keys() and data_entry["need_reproduce"]==False:
        return data_entry
    prompt = data_entry["prompt"]
    test_case_0 = data_entry["test_list"][0]
    function_name = test_case_0.split("(")[0].split(" ")[-1]

    text = f"""
{construct_few_shot_prompt}

**Input Code Snippet**:
```python
{prompt}
```
"""
    test_case_list = []
    for i in range(times):
        while True:
            try:
                message = client.messages.create(
                    model="claude-3-5-sonnet-20241022",
                    max_tokens=4096,
                    system="You are a code developer assistant.",
                    messages=[
                        {"role": "user", "content": text}
                    ]
                )
                test_case = message.content[0].text
                test_case = preprocess_data(test_case)
            except Exception as e:
                time.sleep(20)
                print(e)
                test_case = ""
            if test_case!="":
                break
        test_case_list.append(test_case)
    data_entry["test_case_list"] = test_case_list
    return data_entry

def call_fetch_test_completion_helper(dataset, model,lg):
    print("Fixing bug...")
    with ThreadPoolExecutor(max_workers=5) as executor:
        future_to_entry = {executor.submit(fetch_completion, copy.deepcopy(entry), model, lg): entry for entry in tqdm(dataset)}
        for future in tqdm(concurrent.futures.as_completed(future_to_entry)):
            entry = future_to_entry[future]
            try:
                updated_entry = future.result()
                idx = dataset.index(entry)
                dataset[idx] = updated_entry
            except Exception as e:
                print(repr(e))
    return dataset


if __name__ == "__main__":
    model_list = ["claude-3-5-sonnet-20241022"]
    language = ["python"]
    for model in model_list:
        for lg in language:
            from datasets import load_dataset
            with open(f"./dataset/{model}_mbpp.json", "r") as f:
                dataset = json.load(f)
            dataset = [entry for entry in dataset]
            with ThreadPoolExecutor(max_workers=5) as executor:
                future_to_entry = {executor.submit(fetch_completion, copy.deepcopy(entry), model, lg): entry for entry in tqdm(dataset)}
                for future in tqdm(concurrent.futures.as_completed(future_to_entry)):
                    entry = future_to_entry[future]
                    try:
                        updated_entry = future.result()
                        idx = dataset.index(entry)
                        dataset[idx] = updated_entry
                    except Exception as e:
                        print(repr(e))

            with open(f"./dataset/{model}_mbpp.json", "w") as f:
                json.dump(dataset, f, indent=4)
