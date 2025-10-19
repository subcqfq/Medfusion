# %%
import pandas as pd
df = pd.read_csv('mimichstest.csv')

# %%
import os
os.environ['OPENAI_API_KEY'] = 'xxxx'


# %%
import numpy as np
from openai import OpenAI
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

SAVE_FILE = 'results.npy'

# Load existing results
try:
    results_dict = dict(np.load(SAVE_FILE, allow_pickle=True).item())
    print(f"Loaded {len(results_dict)} historical records")
except FileNotFoundError:
    results_dict = {}
    print("No previous data found; starting from scratch")

def save_progress():
    """Save results immediately"""
    np.save(SAVE_FILE, np.array(results_dict, dtype=object))
    print(f"Progress saved: {len(results_dict)} items")

def get_response(user_input):
    client = OpenAI(
        api_key=os.getenv('OPENAI_API_KEY'),  # If you haven't set the environment variable, replace with "your-api-key"
        base_url="xxxxx",  # This uses Alibaba Cloud's LLM; for other platforms, consult their docs and modify accordingly.
    )
    messages = [
        {'role': 'assistant', 'content': 'You are a medical expert'},
        {'role': 'user', 'content': user_input}
    ]
    completion = client.chat.completions.create(
        model="qwen-max",
        messages=messages
    )
    messages.append({'role': 'assistant', 'content': completion.choices[0].message.content})
    messages.append({'role': 'user', 'content': 'Output only a single numeric value indicating the probability of death within the next 24 hours, without a percent sign, and nothing else.'})
    completion = client.chat.completions.create(
        model="qwen-max",  # deepseek-r1 is used here as an example; you can change the model name as needed.
        messages=messages
    )
    print("="*20 + "Final Answer" + "="*20)
    print(completion.choices[0].message.content)
    return completion.choices[0].message.content

def build_prompt(i):
    gender = "male" if round(df.loc[i, 'gender'], 2) == 1 else "female"
    age = round(df.loc[i, 'age'], 2)
    heart_rate = round(df.loc[i, 'heart_rate'], 2)
    mbp = round(df.loc[i, 'mbp'], 2)
    temperature = round(df.loc[i, 'temperature'], 2)
    spo2 = round(df.loc[i, 'spo2'], 2)
    resp_rate = round(df.loc[i, 'resp_rate'], 2)
    pao2 = round(df.loc[i, 'pao2'], 2)
    wbc = round(df.loc[i, 'wbc'], 2)
    creatinine = round(df.loc[i, 'creatinine'], 2)
    bun = round(df.loc[i, 'bun'], 2)
    sodium = round(df.loc[i, 'sodium'], 2)
    albumin = round(df.loc[i, 'albumin'], 2)
    bilirubin = round(df.loc[i, 'bilirubin'], 2)
    glucose = round(df.loc[i, 'glucose'], 2)
    ph = round(df.loc[i, 'ph'], 2)
    pco2 = round(df.loc[i, 'pco2'], 2)
    gcs = round(df.loc[i, 'gcs'], 2)
    pao2fio2ratio = round(df.loc[i, 'pao2fio2ratio'], 2)
    platelet = round(df.loc[i, 'platelet'], 2)
    pt = round(df.loc[i, 'pt'], 2)
    potassium = round(df.loc[i, 'potassium'], 2)
    gcs_motor = round(df.loc[i, 'gcs_motor'], 2)
    gcs_verbal = round(df.loc[i, 'gcs_verbal'], 2)
    gcs_eyes = round(df.loc[i, 'gcs_eyes'], 2)
    alt = round(df.loc[i, 'alt'], 2)
    ast = round(df.loc[i, 'ast'], 2)
    baseexcess = round(df.loc[i, 'baseexcess'], 2)
    totalco2 = round(df.loc[i, 'totalco2'], 2)
    lactate = round(df.loc[i, 'lactate'], 2)
    free_calcium = round(df.loc[i, 'free_calcium'], 2)
    fio2 = round(df.loc[i, 'fio2'], 2)
    sbp = round(df.loc[i, 'sbp'], 2)
    dbp = round(df.loc[i, 'dbp'], 2)
    print(df.loc[i, 'death'])

    return (
        f"Here is a {age}-year-old {gender} patient: heart rate {heart_rate} bpm, "
        f"mean arterial pressure {mbp} mmHg, systolic blood pressure {sbp} mmHg, "
        f"diastolic blood pressure {dbp} mmHg, body temperature {temperature} °C, "
        f"oxygen saturation {spo2}%, respiratory rate {resp_rate} bpm, "
        f"arterial oxygen partial pressure (PaO2) {pao2} mmHg, white blood cell count {wbc} K/uL, "
        f"creatinine {creatinine} mg/dL, blood urea nitrogen {bun} mg/dL, serum sodium {sodium} mEq/L, "
        f"serum albumin {albumin} g/dL, bilirubin {bilirubin} mg/dL, glucose {glucose} mg/dL, "
        f"pH value {ph}, arterial carbon dioxide partial pressure (PaCO2) {pco2} mmHg, "
        f"GCS score {gcs}, GCS motor {gcs_motor}, GCS verbal {gcs_verbal}, GCS eyes {gcs_eyes}, "
        f"oxygenation index (PaO2/FiO2) {pao2fio2ratio}, platelet count {platelet} K/uL, prothrombin time {pt} seconds, "
        f"serum potassium {potassium} mEq/L, alanine aminotransferase (ALT) {alt} IU/L, "
        f"aspartate aminotransferase (AST) {ast} IU/L, base excess {baseexcess} mEq/L, "
        f"total CO2 {totalco2} mEq/L, lactate {lactate} mmol/L, ionized calcium {free_calcium} mmol/L, "
        f"fraction of inspired oxygen (FiO2) {fio2}%. "
         f"What is this patient's probability of developing hemorrhagic shock within the next 6 hours? "
        f"Provide a detailed analysis process, and you must also provide the probability."
    )

def process_row(i):
    user_input = build_prompt(i)
    result = get_response(user_input)
    results_dict[i] = result
    save_progress()  # Save immediately after each task completes
    return i, result

if __name__ == "__main__":
    workers = 130  # Parallelism per batch
    processed_set = set(results_dict.keys())  # Processed row indices

    for batch_start in range(0, df.shape[0], workers):
        batch_indices = [i for i in range(batch_start, min(batch_start + workers, df.shape[0])) if i not in processed_set]
        if not batch_indices:
            continue  # This batch has already been processed; skip

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(process_row, i) for i in batch_indices]
            for future in as_completed(futures):
                i, result = future.result() 
                # results_dict[i] = result
                # save_progress()  # Save immediately after each task completes

        print(f"Completed {batch_indices}, waiting 6 seconds...")
        time.sleep(6)

    print("All done ✅")


# %%
import numpy as np
import pandas as pd

# Read the .npy file
modsp = np.load('llama_results.npy', allow_pickle=True).item()  # Extract the dictionary

# Convert to DataFrame
df = pd.DataFrame(list(modsp.items()), columns=['key', 'value'])
modsp
