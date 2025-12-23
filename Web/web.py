import os
os.environ['OPENAI_API_KEY'] = 'xxxxxx'

import os
import pandas as pd
import joblib
import shap
import numpy as np
import json
from openai import OpenAI
from typing import Dict, List, Callable, Any
import gradio as gr
from shape_print import explain_mods_kernel_shap

# Base class for tools and a simple tool registry
class Tool:
    def __init__(self, name: str, description: str, func: Callable, params_description: str = ""):
        self.name = name
        self.description = description
        self.func = func
        self.params_description = params_description

    def run(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    def register_tool(self, tool: Tool):
        self.tools[tool.name] = tool
    def get_tool(self, name: str) -> Tool:
        return self.tools.get(name)
    def list_tools(self) -> List[Dict[str, str]]:
        return [{"name": t.name, "description": t.description} for t in self.tools.values()]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def patientdata_tool(**patient_data) -> str:
    """Obtain patient data from user input."""
    try:
        gender = "male" if patient_data.get('gender') == "Male" else "female"
        age = patient_data.get('age', 0)
        heart_rate = patient_data.get('heart_rate', 0)
        mbp = patient_data.get('mbp', 0)
        temperature = patient_data.get('temperature', 0)
        spo2 = patient_data.get('spo2', 0)
        resp_rate = patient_data.get('resp_rate', 0)
        pao2 = patient_data.get('pao2', 0)
        wbc = patient_data.get('wbc', 0)
        creatinine = patient_data.get('creatinine', 0)
        bun = patient_data.get('bun', 0)
        sodium = patient_data.get('sodium', 0)
        albumin = patient_data.get('albumin', 0)
        bilirubin = patient_data.get('bilirubin', 0)
        glucose = patient_data.get('glucose', 0)
        ph = patient_data.get('ph', 0)
        pco2 = patient_data.get('pco2', 0)
        gcs = patient_data.get('gcs', 0)
        pao2fio2ratio = patient_data.get('pao2fio2ratio', 0)
        platelet = patient_data.get('platelet', 0)
        pt = patient_data.get('pt', 0)
        potassium = patient_data.get('potassium', 0)
        gcs_motor = patient_data.get('gcs_motor', 0)
        gcs_verbal = patient_data.get('gcs_verbal', 0)
        gcs_eyes = patient_data.get('gcs_eyes', 0)
        alt = patient_data.get('alt', 0)
        ast = patient_data.get('ast', 0)
        baseexcess = patient_data.get('baseexcess', 0)
        totalco2 = patient_data.get('totalco2', 0)
        lactate = patient_data.get('lactate', 0)
        free_calcium = patient_data.get('free_calcium', 0)
        fio2 = patient_data.get('fio2', 0)
        sbp = patient_data.get('sbp', 0)
        dbp = patient_data.get('dbp', 0)

        return (
            f"A {age}-year-old {gender} patient: heart rate {heart_rate} bpm, "
            f"mean arterial pressure {mbp} mmHg, systolic BP {sbp} mmHg, diastolic BP {dbp} mmHg, "
            f"temperature {temperature} °C, SpO₂ {spo2}%, respiratory rate {resp_rate} bpm, "
            f"PaO₂ {pao2} mmHg, WBC {wbc} K/uL, creatinine {creatinine} mg/dL, BUN {bun} mg/dL, "
            f"serum sodium {sodium} mEq/L, albumin {albumin} g/dL, bilirubin {bilirubin} mg/dL, "
            f"glucose {glucose} mg/dL, pH {ph}, PaCO₂ {pco2} mmHg, GCS {gcs} "
            f"(motor {gcs_motor}, verbal {gcs_verbal}, eye {gcs_eyes}), "
            f"PaO₂/FiO₂ ratio {pao2fio2ratio}, platelets {platelet} K/uL, PT {pt} s, "
            f"serum potassium {potassium} mEq/L, ALT {alt} IU/L, AST {ast} IU/L, "
            f"base excess {baseexcess} mEq/L, total CO₂ {totalco2} mEq/L, lactate {lactate} mmol/L, "
            f"ionized calcium {free_calcium} mmol/L, FiO₂ {fio2}%"
        )
    except Exception as e:
        return f"Error parsing patient data: {str(e)}"

def shap_to_text_prob(explainer, shap_values, x_single, model):
    base_logit = explainer.expected_value
    base_prob = sigmoid(base_logit)
    scaler = joblib.load("scaler.pkl")
    X_example = scaler.transform(x_single.values.reshape(1, -1))
    pred_prob = model.predict_proba(X_example)[0, 1]

    contributions = []
    logit_running = base_logit
    for feature, value, shap_val in zip(x_single.index, x_single.values, shap_values):
        prob_without = sigmoid(logit_running)
        prob_with = sigmoid(logit_running + shap_val)
        delta_prob = (prob_with - prob_without) * 100
        logit_running += shap_val
        contributions.append((feature, value, delta_prob))

    contributions_sorted = sorted(contributions, key=lambda x: abs(x[2]), reverse=True)
    explanations = []
    for feature, value, delta_prob in contributions_sorted[:10]:
        if delta_prob > 0:
            explanations.append(f"{feature} = {value} increased disease risk by {abs(delta_prob):.1f}%")
        else:
            explanations.append(f"{feature} = {value} decreased disease risk by {abs(delta_prob):.1f}%")
    return explanations, pred_prob, base_prob

def create_dataframe_from_inputs(**patient_data):
    """Create a DataFrame from user inputs."""
    gender_value = 1 if patient_data.get('gender') == "Male" else 0
    data = {
        'gender': [gender_value],
        'age': [patient_data.get('age', 0)],
        'heart_rate': [patient_data.get('heart_rate', 0)],
        'mbp': [patient_data.get('mbp', 0)],
        'temperature': [patient_data.get('temperature', 0)],
        'spo2': [patient_data.get('spo2', 0)],
        'resp_rate': [patient_data.get('resp_rate', 0)],
        'pao2': [patient_data.get('pao2', 0)],
        'wbc': [patient_data.get('wbc', 0)],
        'creatinine': [patient_data.get('creatinine', 0)],
        'bun': [patient_data.get('bun', 0)],
        'sodium': [patient_data.get('sodium', 0)],
        'albumin': [patient_data.get('albumin', 0)],
        'bilirubin': [patient_data.get('bilirubin', 0)],
        'glucose': [patient_data.get('glucose', 0)],
        'ph': [patient_data.get('ph', 0)],
        'pco2': [patient_data.get('pco2', 0)],
        'gcs': [patient_data.get('gcs', 0)],
        'pao2fio2ratio': [patient_data.get('pao2fio2ratio', 0)],
        'platelet': [patient_data.get('platelet', 0)],
        'pt': [patient_data.get('pt', 0)],
        'potassium': [patient_data.get('potassium', 0)],
        'gcs_motor': [patient_data.get('gcs_motor', 0)],
        'gcs_verbal': [patient_data.get('gcs_verbal', 0)],
        'gcs_eyes': [patient_data.get('gcs_eyes', 0)],
        'alt': [patient_data.get('alt', 0)],
        'ast': [patient_data.get('ast', 0)],
        'baseexcess': [patient_data.get('baseexcess', 0)],
        'totalco2': [patient_data.get('totalco2', 0)],
        'lactate': [patient_data.get('lactate', 0)],
        'free_calcium': [patient_data.get('free_calcium', 0)],
        'fio2': [patient_data.get('fio2', 0)],
        'sbp': [patient_data.get('sbp', 0)],
        'dbp': [patient_data.get('dbp', 0)]
    }
    return pd.DataFrame(data)

def Dead_analysis(**patient_data) -> str:
    try:
        df = create_dataframe_from_inputs(**patient_data)
        output = []
        output.append(patientdata_tool(**patient_data))
        output.append("Mortality model analysis results:")
        analysis_value = explain_mods_kernel_shap(
            df_patient=df,
            df_train="./Dead/mimicdeadtrain-v2.csv",
            scaler_file="./Dead/dead_scaler_v1.pkl",
            model_file="./Dead/dead_xgb_model_v2.pkl",
            label_col=None,
            patient_row=0,
            background_size=100,
            nsamples="auto"
        )
        output.append(f"{analysis_value}")
        output.append("Based on SHAP analysis, the patient's 24-hour mortality risk assessment is complete.")
        output.append("present the SHAP analysis results in detail in a tabular format, Add Clinical Interpretation in the table to determine whether each SHAP value aligns with clinical.")
        return "\n".join(output)
    except Exception as e:
        return f"Dead model analysis error: {str(e)}"
    
def Hx_analysis(**patient_data) -> str:
    try:
        df = create_dataframe_from_inputs(**patient_data)
        output = []
        output.append(patientdata_tool(**patient_data))
        output.append("Hypoxemia model analysis results:")
        analysis_value = explain_mods_kernel_shap(
            df_patient=df,
            f_train="./Hx/mimichxtrain-v1.csv",
            scaler_file="./HX/HX_scaler.pkl",
            model_file="./HX/HX_XGB_Model.pkl",
            label_col=None,
            patient_row=0,
            background_size=100,
            nsamples="auto"
        )
        output.append(f"{analysis_value}")
        output.append("Based on SHAP analysis, the patient's 6-hour hypoxemia risk assessment is complete.")
        output.append("present the SHAP analysis results in detail in a tabular format, Add Clinical Interpretation in the table to determine whether each SHAP value aligns with clinical.")
        return "\n".join(output)
    except Exception as e:
        return f"HX model analysis error: {str(e)}"

def Hs_analysis(**patient_data) -> str:
    try:
        df = create_dataframe_from_inputs(**patient_data)
        output = []
        output.append(patientdata_tool(**patient_data))
        output.append("Hemorrhagic shock model analysis results:")
        analysis_value = explain_mods_kernel_shap(
            df_patient=df,
            df_train="./Hs/mimichstrain-v1.csv",
            scaler_file="./Hs/Hs_scaler.pkl",
            model_file="./Hs/hs_xgb_best_model.pkl",
            label_col=None,
            patient_row=0,
            background_size=100,
            nsamples="auto"
        )
        output.append(f"{analysis_value}")
        output.append("Based on SHAP analysis, the patient's 6-hour hemorrhagic shock risk assessment is complete.")
        output.append("present the SHAP analysis results in detail in a tabular format, Add Clinical Interpretation in the table to determine whether each SHAP value aligns with clinical.")
        return "\n".join(output)
    except Exception as e:
        return f"HS model analysis error: {str(e)}"

def mods_analysis(**patient_data) -> str:
    try:
        df = create_dataframe_from_inputs(**patient_data)
        X_origin = df.iloc[:, :]
        scaler = joblib.load("scaler.pkl")
        X_scaled = scaler.transform(X_origin)
        model = joblib.load("xgb.pkl")
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)
        explanations, pred_prob, base_prob = shap_to_text_prob(
            explainer, shap_values[0], X_origin.iloc[0], model
        )
        output = []
        output.append(patientdata_tool(**patient_data))
        output.append("MODS model analysis results:")
        output.append(f"Baseline average risk: {base_prob:.2%}")
        output.append(f"Predicted risk for this sample: {pred_prob:.2%}")
        output.append("Top 10 contributing features:")
        output.extend(explanations)
        output.append("Based on SHAP analysis, the patient's 24-hour MODS risk assessment is complete.")
        output.append("present the SHAP analysis results in detail in a tabular format, Add Clinical Interpretation in the table to determine whether each SHAP value aligns with clinical.")
        return "\n".join(output)
    except Exception as e:
        return f"MODS model analysis error: {str(e)}"

def choose_tools(user_input: str, available_tools: List[Dict[str, str]]) -> List[str]:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
    tools_info = "\n".join([f"Tool name: {tool['name']}\nDescription: {tool['description']}" for tool in available_tools])
    prompt = (
        f"Available tools:\n{tools_info}\n\n"
        f"User request: {user_input}\n\n"
        "Analyze the user's request and choose which tools to call. "
        "If multiple tools are needed, list them all. "
        "Return ONLY a JSON array of tool names, e.g.: ['tool1', 'tool2']"
    )
    messages = [
        {'role': 'system', 'content': 'You are an expert at selecting tools based on the user\'s question.'},
        {'role': 'user', 'content': prompt}
    ]
    try:
        completion = client.chat.completions.create(
            model="llama-4-maverick-17b-128e-instruct",
            messages=messages,
            max_tokens=512
        )
        response = completion.choices[0].message.content
        tool_names = json.loads(response)
        return tool_names if isinstance(tool_names, list) else []
    except Exception:
        return [tool['name'] for tool in available_tools]

def get_comprehensive_analysis(user_input: str, tool_results: List[str], conversation_history: List[Dict[str, str]]) -> tuple:
    client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'),
                    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")
 
    # Create a copy of conversation_history to avoid modifying the original data
    updated_history = conversation_history.copy() if conversation_history else []
    
    combined_results = "\n\n".join(tool_results)
    prompt_1 = f"User question: {user_input}\n"
    prompt_2 = f"{combined_results}"
    
    messages = updated_history + [{'role': 'user', 'content': prompt_1}]
    messages += [{'role': 'assistant', 'content': prompt_2}]
    
    # Update the history
    updated_history.append({'role': 'user', 'content': prompt_1})
    updated_history.append({'role': 'assistant', 'content': prompt_2})

    try:
        completion = client.chat.completions.create(
            model="llama-4-maverick-17b-128e-instruct",
            messages=messages,
            max_tokens=4096,
            temperature=0.6
        )
        answer = completion.choices[0].message.content
        updated_history.append({"role": "assistant", "content": answer})
        return answer, updated_history
    except Exception as e:
        return f"Comprehensive analysis error: {str(e)}", updated_history

def respond(message, chat_history, conversation_state, *input_values):

    # Initialize state
    if chat_history is None:
        chat_history = []
    if conversation_state is None:
        conversation_state = []

    # Process input data
    input_names = [
        'gender', 'age', 'heart_rate', 'sbp', 'dbp', 'mbp', 'temperature', 'spo2', 
        'resp_rate', 'pao2', 'fio2', 'pao2fio2ratio', 'wbc', 'platelet', 'creatinine', 
        'bun', 'sodium', 'potassium', 'albumin', 'bilirubin', 'glucose', 'alt', 'ast', 
        'ph', 'pco2', 'baseexcess', 'totalco2', 'lactate', 'free_calcium', 'gcs', 
        'gcs_motor', 'gcs_verbal', 'gcs_eyes', 'pt'
    ]
    patient_data = dict(zip(input_names, input_values))

    # Register tools
    tool_registry = ToolRegistry()
    tool_registry.register_tool(Tool(
        name="patient_data_parser",
        description="Parse the patient's basic vital signs and laboratory data.",
        func=lambda: patientdata_tool(**patient_data)
    ))
    tool_registry.register_tool(Tool(
        name="mods_risk_analyzer",
        description="Analyze the patient's risk of MODS in the next 24 hours. Use only if the prompt explicitly mentions MODS and 'next 24 hours'.",
        func=lambda: mods_analysis(**patient_data)
    ))
    tool_registry.register_tool(Tool(
        name="Hx_risk_analyzer",
        description="Analyze the patient's risk of hypoxemia in the next 6 hours. Use only if the prompt explicitly mentions hypoxemia and 'next 6 hours'.",
        func=lambda: Hx_analysis(**patient_data)
    ))
    tool_registry.register_tool(Tool(
        name="Hs_risk_analyzer",
        description="Analyze the patient's risk of hemorrhagic shock in the next 6 hours. Use only if the prompt explicitly mentions hemorrhagic shock and 'next 6 hours'.",
        func=lambda: Hs_analysis(**patient_data)
    ))
    tool_registry.register_tool(Tool(
        name="Dead_risk_analyzer",
        description="Analyze the patient's risk of mortality in the next 24 hours. Use only if the prompt explicitly mentions death/mortality and 'next 24 hours'.",
        func=lambda: Dead_analysis(**patient_data)
    ))
    
    available_tools = tool_registry.list_tools()

    # Select and execute tools
    selected_tools = choose_tools(message, available_tools)
    tool_results = []
    for tool_name in selected_tools:
        tool = tool_registry.get_tool(tool_name)
        if tool:
            try:
                result = tool.run()
                tool_results.append(f"{result}")
            except Exception as e:
                tool_results.append(f"=== {tool.name} execution error ===\n{e}")

    # Get comprehensive analysis and pass conversation_state
    answer, updated_conversation_state = get_comprehensive_analysis(message, tool_results, conversation_state)
    
    # Update chat UI
    chat_history.append((message, answer))
    
    # Return an empty input box, updated chat history, and updated conversation state
    return "", chat_history, updated_conversation_state

def clear_conversation():
    """Function to clear the conversation history."""
    return None, []  # Clear chatbot and conversation_state

# === CSS styles ===
custom_css = """
/* Base style reset */
.gradio-container {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
    max-width: 100% !important;
    margin: 0 auto !important;
    padding: 15px !important;
    background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    min-height: 100vh;
}

.full-bleed {
    width: 100vw;
    margin-left: calc(50% - 50vw);
    margin-right: calc(50% - 50vw);
    padding-left: 12px;
    padding-right: 12px;
}

html, body, gradio-app, #root, .gradio-container {
    width: 100%;
    max-width: 100%;
    overflow-x: hidden;
}

/* Main header area */
.main-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 20px;
    border-radius: 20px;
    margin-bottom: 8px;
    text-align: center;
    color: white;
    box-shadow: 0 10px 40px rgba(102,126,234,0.3);
    position: relative;
    overflow: hidden;
}

.main-header::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(45deg, rgba(255,255,255,0.1) 0%, transparent 50%);
    pointer-events: none;
}

.main-title {
    font-size: 2.6em;
    font-weight: 800;
    margin: 0 0 10px 0;
    text-shadow: 0 3px 8px rgba(0,0,0,0.3);
    background: linear-gradient(45deg, #fff, #f0f8ff);
    background-clip: text;
    -webkit-background-clip: text;
    position: relative;
    z-index: 1;
}

.main-subtitle {
    font-size: 1.1em;
    opacity: 0.95;
    margin: 0;
    font-weight: 400;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
    position: relative;
    z-index: 1;
}

/* ============ Feature showcase area ============ */
.feature-showcase {
    margin: 5px auto;
    padding: 10px;
    max-width: 1200px;
    width: 100%;
}

.feature-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 20px;
    padding: 10px 0;
    max-width: 100%;
    margin: 0 auto;
}

.feature-card {
    background: rgba(255, 255, 255, 0.95);
    border-radius: 16px;
    padding: 18px 15px;
    text-align: center;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
    transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    border: 1px solid rgba(255, 255, 255, 0.3);
    backdrop-filter: blur(10px);
    position: relative;
    overflow: hidden;
    min-height: 180px;
    display: flex;
    flex-direction: column;
    justify-content: center;
}

feature-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    height: 4px;
    background: linear-gradient(90deg, #667eea, #764ba2, #f093fb, #f5576c);
    transform: translateX(-100%);
    transition: transform 0.6s ease;
}

.feature-card:hover::before {
    transform: translateX(0);
}

.feature-card:hover {
    transform: translateY(-8px) scale(1.02);
    box-shadow: 0 15px 50px rgba(0, 0, 0, 0.15);
    background: rgba(255, 255, 255, 1);
}

.feature-icon {
    font-size: 2.8em;
    margin-bottom: 12px;
    display: block;
    filter: drop-shadow(0 4px 8px rgba(0, 0, 0, 0.1));
    transition: all 0.3s ease;
}

.feature-card:hover .feature-icon {
    transform: scale(1.1) rotate(5deg);
    filter: drop-shadow(0 6px 12px rgba(0, 0, 0, 0.2));
}

.feature-title {
    font-size: 1.25em;
    font-weight: 700;
    margin: 0 0 10px 0;
    color: #2d3748;
    background: linear-gradient(135deg, #667eea, #764ba2);
    background-clip: text;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    transition: all 0.3s ease;
}

.feature-card:hover .feature-title {
    transform: translateY(-2px);
}

.feature-desc {
    font-size: 0.9em;
    color: #718096;
    line-height: 1.5;
    margin: 0;
    transition: color 0.3s ease;
}

.feature-card:hover .feature-desc {
    color: #4a5568;
}

/* Decorative shimmer effect */
@keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

.feature-card:hover::after {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
    animation: shimmer 1.5s ease-out;
    pointer-events: none;
}

/* Data input area */
.compact-section {
    background: rgba(255,255,255,0.98);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 15px;
    box-shadow: 0 4px 20px rgba(0,0,0,0.08);
    backdrop-filter: blur(15px);
    transition: all 0.3s ease;
}

.compact-section:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(0,0,0,0.12);
}

.section-header {
    font-size: 1.2em;
    font-weight: 700;
    color: #2d3748;
    margin: 0 0 18px 0;
    padding-bottom: 10px;
    border-bottom: 3px solid transparent;
    border-image: linear-gradient(90deg, #667eea, #764ba2) 1;
    display: flex;
    align-items: center;
}

.section-icon {
    margin-right: 12px;
    font-size: 1.4em;
    filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
}

/* Chat area */
.chat-container {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 10px 40px rgba(240,147,251,0.3);
    height: calc(100vh - 200px);
    min-height: 700px;
    display: flex;
    flex-direction: column;
    position: relative;
    overflow: hidden;
}

.chat-container::before {
    content: '';
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background: radial-gradient(circle at 20% 80%, rgba(255,255,255,0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(255,255,255,0.1) 0%, transparent 50%);
    pointer-events: none;
}

.chat-header {
    color: white;
    font-size: 1.6em;
    font-weight: 700;
    text-align: center;
    margin-bottom: 25px;
    text-shadow: 0 2px 8px rgba(0,0,0,0.3);
    position: relative;
    z-index: 1;
}

/* Main layout */
#main-row {
    display: flex;
    align-items: flex-start;
    gap: 25px;
    margin-top: 8px;
}

#left-pane {
    flex: 0 0 400px;
    max-width: 400px;
    min-width: 380px;
    height: calc(100vh - 200px);
    overflow-y: auto;
    padding-right: 15px;
}

#right-pane {
    flex: 1;
    min-width: 0;
}

/* Input controls */
.gradio-textbox, .gradio-number, .gradio-radio {
    border-radius: 12px !important;
    border: 2px solid rgba(102,126,234,0.2) !important;
    font-size: 0.95em !important;
    transition: all 0.3s ease !important;
    background: rgba(255,255,255,0.9) !important;
}

.gradio-textbox:focus, .gradio-number:focus {
    border-color: #667eea !important;
    box-shadow: 0 0 0 4px rgba(102,126,234,0.15) !important;
    background: rgba(255,255,255,1) !important;
}

.gradio-chatbot {
    border-radius: 16px !important;
    border: none !important;
    background: rgba(255,255,255,0.98) !important;
    backdrop-filter: blur(15px) !important;
    flex: 1 !important;
    min-height: 450px !important;
    box-shadow: 0 8px 32px rgba(0,0,0,0.1) !important;
}

.gradio-label {
    color: #374151 !important;
    font-weight: 600 !important;
    font-size: 0.9em !important;
    margin-bottom: 6px !important;
}

/* Buttons */
#chat-controls .send-btn button,
#chat-controls .send-btn .gr-button {
    width: 100%;
    padding: 12px 20px;
    min-height: 0;
    font-size: 1em;
    font-weight: 600;
    background: linear-gradient(135deg, #667eea, #764ba2);
    border: none;
    border-radius: 12px;
    color: white;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px rgba(102,126,234,0.3);
}

#chat-controls .send-btn button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(102,126,234,0.4);
}

.quick-row {
    display: flex;
    gap: 12px;
    flex-wrap: nowrap;
    overflow-x: auto;
    margin-top: 15px;
}

.quick-btn button,
.quick-btn .gr-button {
    padding: 8px 16px;
    font-size: 0.9em;
    line-height: 1.3;
    white-space: nowrap;
    background: rgba(255,255,255,0.9);
    color: #667eea;
    border: 2px solid rgba(102,126,234,0.2);
    border-radius: 10px;
    transition: all 0.3s ease;
}

.quick-btn button:hover {
    background: #667eea;
    color: white;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(102,126,234,0.3);
}

.quick-btn {
    flex: 1 1 auto;
    min-width: 0;
}

.clear-btn button {
    padding: 8px 16px;
    font-size: 0.9em;
    background: #ef4444;
    color: white;
    border: none;
    border-radius: 10px;
    transition: all 0.3s ease;
    margin-left: 8px;
}

.clear-btn button:hover {
    background: #dc2626;
    transform: translateY(-1px);
    box-shadow: 0 4px 12px rgba(239,68,68,0.3);
}

/* Scrollbar */
#left-pane::-webkit-scrollbar {
    width: 8px;
}

#left-pane::-webkit-scrollbar-track {
    background: rgba(0,0,0,0.05);
    border-radius: 10px;
}

#left-pane::-webkit-scrollbar-thumb {
    background: linear-gradient(135deg, #667eea, #764ba2);
    border-radius: 10px;
}

#left-pane::-webkit-scrollbar-thumb:hover {
    background: linear-gradient(135deg, #5a6fd8, #6a42a0);
}

/* Responsive */
@media (max-width: 1200px) {
    .feature-grid {
        grid-template-columns: repeat(2, 1fr);
        gap: 15px;
    }
    
    .feature-showcase {
        margin: 6px auto;
    }
    
    #main-row {
        flex-direction: column;
        margin-top: 6px;
    }
    
    #left-pane {
        flex: none;
        max-width: 100%;
        height: auto;
        margin-bottom: 15px;
    }
}

@media (max-width: 768px) {
    .feature-grid {
        grid-template-columns: 1fr;
        gap: 10px;
    }
    
    .feature-showcase {
        margin: 5px auto;
        padding: 5px;
    }
    
    #main-row {
        margin-top: 5px;
    }
    
    .main-header {
        padding: 15px;
        margin-bottom: 5px;
    }
}
"""

# Build UI
with gr.Blocks(css=custom_css, title="Intelligent Assessment System", theme=gr.themes.Soft()) as demo:
    # Add a State component to store the conversation history
    conversation_state = gr.State([])
    
    # Header (full width)
    gr.HTML("""
        <div class="main-header">
            <div class="main-title">🏥 Intelligent Disease Risk Assessment System</div>
            <div class="main-subtitle">Disease risk prediction powered by machine learning and large language models</div>
        </div>
    """, elem_classes="full-bleed")
    
    # Feature showcase
    with gr.Row(elem_classes="feature-showcase"):
        gr.HTML("""
            <div class="feature-grid">
                <div class="feature-card">
                    <div class="feature-icon">🤖</div>
                    <div class="feature-title">Smart analysis</div>
                    <div class="feature-desc">Machine learning models with SHAP explainability to deliver precise risk assessments</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">📊</div>
                    <div class="feature-title">Risk assessment</div>
                    <div class="feature-desc">Predict multiple disease risks with quantitative metrics and risk stratification</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">💬</div>
                    <div class="feature-title">Conversational QA</div>
                    <div class="feature-desc">Natural-language interaction with multi-turn dialogue and personalized medical Q&A</div>
                </div>
                <div class="feature-card">
                    <div class="feature-icon">🔍</div>
                    <div class="feature-title">Deep explanations</div>
                    <div class="feature-desc">Detailed analysis of how each physiological metric contributes to risk</div>
                </div>
            </div>
        """)

    # Main content (full width, single row)
    with gr.Row(elem_id="main-row", elem_classes="full-bleed"):
        # Left: patient inputs (compact)
        with gr.Column(scale=2, min_width=400, elem_id="left-pane"):
            # Demographics & Vitals
            with gr.Group(elem_classes="compact-section"):
                gr.HTML('<div class="section-header"><span class="section-icon">👤</span>Demographics & Vitals</div>')
                with gr.Row():
                    gender = gr.Radio(["Male", "Female"], label="Gender", value="Male")
                    age = gr.Number(label="Age", value=65, minimum=0, maximum=120)
                with gr.Row():
                    heart_rate = gr.Number(label="Heart rate (bpm)", value=80, minimum=0)
                    temperature = gr.Number(label="Temperature (°C)", value=36.5)
                with gr.Row():
                    sbp = gr.Number(label="Systolic BP (mmHg)", value=120, minimum=0)
                    dbp = gr.Number(label="Diastolic BP (mmHg)", value=80, minimum=0)
                with gr.Row():
                    mbp = gr.Number(label="Mean arterial pressure (mmHg)", value=93, minimum=0)
                    resp_rate = gr.Number(label="Respiratory rate (bpm)", value=16, minimum=0)
            
            # Oxygenation & Acid–Base
            with gr.Group(elem_classes="compact-section"):
                gr.HTML('<div class="section-header"><span class="section-icon">🫁</span>Oxygenation & Acid–Base</div>')
                with gr.Row():
                    spo2 = gr.Number(label="SpO₂ (%)", value=98)
                    pao2 = gr.Number(label="PaO₂ (mmHg)", value=90)
                with gr.Row():
                    fio2 = gr.Number(label="FiO₂ (%)", value=21)
                    pao2fio2ratio = gr.Number(label="PaO₂/FiO₂ ratio", value=400)
                with gr.Row():
                    ph = gr.Number(label="pH", value=7.4)
                    pco2 = gr.Number(label="PaCO₂ (mmHg)", value=40)
                with gr.Row():
                    baseexcess = gr.Number(label="Base excess (mEq/L)", value=0)
                    totalco2 = gr.Number(label="Total CO₂ (mEq/L)", value=24)
            
            # Neurologic assessment
            with gr.Group(elem_classes="compact-section"):
                gr.HTML('<div class="section-header"><span class="section-icon">🧠</span>Neurologic assessment</div>')
                with gr.Row():
                    gcs = gr.Number(label="GCS total", value=15, minimum=3, maximum=15)
                    gcs_motor = gr.Number(label="GCS motor", value=6, minimum=1, maximum=6)
                with gr.Row():
                    gcs_verbal = gr.Number(label="GCS verbal", value=5, minimum=1, maximum=5)
                    gcs_eyes = gr.Number(label="GCS eye opening", value=4, minimum=1, maximum=4)
            
            # Laboratory tests
            with gr.Group(elem_classes="compact-section"):
                gr.HTML('<div class="section-header"><span class="section-icon">🧪</span>Laboratory tests</div>')
                with gr.Row():
                    wbc = gr.Number(label="WBC (K/uL)", value=7)
                    platelet = gr.Number(label="Platelets (K/uL)", value=250)
                with gr.Row():
                    pt = gr.Number(label="Prothrombin time (s)", value=12)
                    creatinine = gr.Number(label="Creatinine (mg/dL)", value=1.0)
                with gr.Row():
                    bun = gr.Number(label="BUN (mg/dL)", value=15)
                    sodium = gr.Number(label="Sodium (mEq/L)", value=140)
                with gr.Row():
                    potassium = gr.Number(label="Potassium (mEq/L)", value=4.0)
                    albumin = gr.Number(label="Albumin (g/dL)", value=3.5)
                with gr.Row():
                    bilirubin = gr.Number(label="Bilirubin (mg/dL)", value=1.0)
                    glucose = gr.Number(label="Glucose (mg/dL)", value=100)
                with gr.Row():
                    lactate = gr.Number(label="Lactate (mmol/L)", value=1.5)
                    free_calcium = gr.Number(label="Ionized calcium (mmol/L)", value=1.2)
                with gr.Row():
                    alt = gr.Number(label="ALT (IU/L)", value=25)
                    ast = gr.Number(label="AST (IU/L)", value=25)
        
        # Right: assistant (stretches to the right edge)
        with gr.Column(scale=5, min_width=600, elem_id="right-pane"):
            with gr.Group(elem_classes="chat-container"):
                gr.HTML('<div class="chat-header">💬 Intelligent Medical Assistant</div>')
                
                chatbot = gr.Chatbot(
                    height=550,
                    show_label=False,
                    avatar_images=("👨‍⚕️", "🤖"),
                    bubble_full_width=False
                )
                
                # Input box + send button (vertical, same width)
                with gr.Column(elem_id="chat-controls"):
                    msg = gr.Textbox(
                        placeholder="💭 Type your question—I'll analyze it using the patient's data...",
                        show_label=False
                    )
                    with gr.Row():
                        submit_btn = gr.Button(
                            "🚀 Send message",
                            variant="primary",
                            size="sm",
                            elem_classes="send-btn",
                            scale=8
                        )
                        clear_btn = gr.Button(
                            "🗑️ Clear",
                            variant="secondary",
                            size="sm",
                            elem_classes="clear-btn",
                            scale=1
                        )
                
                # Four example buttons (single row, compact)
                with gr.Row(elem_classes="quick-row"):
                    example_btn1 = gr.Button("📊 MODS risk analysis", size="sm", variant="secondary", elem_classes="quick-btn")
                    example_btn2 = gr.Button("🔍 Explain risk factors", size="sm", variant="secondary", elem_classes="quick-btn")
                    example_btn3 = gr.Button("📈 Patient data overview", size="sm", variant="secondary", elem_classes="quick-btn")
                    example_btn4 = gr.Button("⚕️ Clinical recommendations", size="sm", variant="secondary", elem_classes="quick-btn")

    # Collect all input components
    all_inputs = [
        gender, age, heart_rate, sbp, dbp, mbp, temperature, spo2, resp_rate, 
        pao2, fio2, pao2fio2ratio, wbc, platelet, creatinine, bun, sodium, 
        potassium, albumin, bilirubin, glucose, alt, ast, ph, pco2, baseexcess, 
        totalco2, lactate, free_calcium, gcs, gcs_motor, gcs_verbal, gcs_eyes, pt
    ]

    # Example button events
    example_btn1.click(lambda: "Please analyze this patient's MODS risk over the next 24 hours", outputs=msg)
    example_btn2.click(lambda: "Please explain the main factors and mechanisms impacting MODS risk in detail", outputs=msg)
    example_btn3.click(lambda: "Please show a full overview of the current patient's physiological data", outputs=msg)
    example_btn4.click(lambda: "Based on the risk assessment, please provide specific clinical advice and interventions", outputs=msg)
    
    # Interactions
    msg.submit(
        respond, 
        [msg, chatbot, conversation_state] + all_inputs, 
        [msg, chatbot, conversation_state]
    )
    submit_btn.click(
        respond, 
        [msg, chatbot, conversation_state] + all_inputs, 
        [msg, chatbot, conversation_state]
    )
    
    # Clear button event
    clear_btn.click(
        clear_conversation,
        outputs=[chatbot, conversation_state]
    )

# Launch app
if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7876,
        share=False,
        show_error=True
    )
