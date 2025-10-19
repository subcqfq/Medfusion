import pandas as pd
import numpy as np
import joblib
import shap
import warnings
from typing import Union, List, Optional

warnings.filterwarnings("ignore")

# =============== 工具函数 ===============
def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))

def ensure_2d_frame(X, columns):
    """把传入的 array/Series 变成 2D DataFrame，并对齐列名。"""
    if isinstance(X, pd.DataFrame):
        # 若已是 DataFrame，则仅保留需要的列（按列名顺序）
        return X[columns]
    if isinstance(X, pd.Series):
        return pd.DataFrame([X.values], columns=columns)
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return pd.DataFrame(X, columns=columns)

def _load_df(maybe_path_or_df: Union[str, pd.DataFrame]) -> pd.DataFrame:
    """既支持 DataFrame 也支持 CSV 路径。"""
    if isinstance(maybe_path_or_df, pd.DataFrame):
        return maybe_path_or_df.copy()
    return pd.read_csv(str(maybe_path_or_df))

# =============== 主函数 ===============
def explain_mods_kernel_shap(
    df_patient: Union[str, pd.DataFrame],
    df_train: Union[str, pd.DataFrame],
    scaler_file: str,
    model_file: str,
    *,
    label_col: Optional[str] = None,
    patient_row: int = 0,
    background_size: int = 100,
    nsamples: Union[str, int] = "auto"
) -> List[str]:
    """
    基于 Kernel SHAP（logit 链接）对单个患者进行解释，返回字符串列表 output。
    
    参数
    ----
    df_patient : str | pd.DataFrame
        患者数据（无标签）。支持 CSV 路径或 DataFrame。若多行，仅解释 patient_row 指定行。
    df_train : str | pd.DataFrame
        训练数据（含标签）。支持 CSV 路径或 DataFrame。
    scaler_file : str
        标准化器 joblib 文件路径。
    model_file : str
        训练好模型的 joblib 文件路径（需要支持 predict_proba）。
    label_col : str | None
        训练集中的标签列名。若为 None，默认取最后一列为标签。
    patient_row : int
        选择解释患者数据的行索引（默认 0）。
    background_size : int
        KernelExplainer 的背景样本数（从训练集前若干行取，默认 100）。
    nsamples : "auto" | int
        shap_values 的采样规模，默认为 "auto"。
    
    返回
    ----
    List[str]
        与原脚本一致的 output（字符串列表）。
    """
    # -------- 读取数据 --------
    df_patient = _load_df(df_patient)
    df_train = _load_df(df_train)

    # 标签列与特征列
    if label_col is None:
        label_col = df_train.columns[-1]
    feature_cols = [c for c in df_train.columns if c != label_col]

    # 选择患者行（只解释一行）
    if not (0 <= patient_row < len(df_patient)):
        raise IndexError(f"patient_row 超出范围：0 ~ {len(df_patient)-1}")
    X_patient_series = df_patient[feature_cols].iloc[patient_row]

    # 背景样本（未标准化的原始特征）
    BACKGROUND_SIZE = max(1, int(background_size))
    background = df_train[feature_cols].head(BACKGROUND_SIZE).copy()

    # -------- 加载标准化器与模型 --------
    scaler = joblib.load(scaler_file)
    model = joblib.load(model_file)

    # 预测函数：输入原始特征 → 标准化 → 返回正类概率
    def model_fn(X):
        X_df = ensure_2d_frame(X, feature_cols)
        X_scaled = scaler.transform(X_df.values)
        return model.predict_proba(X_scaled)[:, 1]

    # 患者预测概率（与 SHAP 使用的同一路径）
    pred_prob = float(model_fn(X_patient_series)[0])

    # -------- 构建 KernelExplainer 并计算 SHAP --------
    # 使用 link='logit'：保证 expected_value（基准 logit）+ Σphi = logit(pred_prob)
    explainer = shap.KernelExplainer(model_fn, background, link="logit")

    # 只解释该患者一行
    shap_values_raw = explainer.shap_values(X_patient_series, nsamples=nsamples)
    # KernelExplainer 对单行返回 1D ndarray（或长度为1的 list/ndarray）
    phi = np.asarray(shap_values_raw).reshape(-1)

    # -------- 文本化解释 --------
    def shap_to_text_prob(explainer_obj, shap_values_1d: np.ndarray, x_single_series: pd.Series):
        """
        把 logit 空间的 SHAP 值近似映射为“概率百分点变化”，并输出 Top 10 解释。
        """
        base_logit = float(np.squeeze(explainer_obj.expected_value))
        base_prob = sigmoid(base_logit)

        contributions = []
        logit_running = base_logit
        for feature, value, shap_val in zip(x_single_series.index, x_single_series.values, shap_values_1d):
            prob_without = sigmoid(logit_running)
            prob_with = sigmoid(logit_running + float(shap_val))
            delta_prob_pct = (prob_with - prob_without) * 100.0
            logit_running += float(shap_val)
            contributions.append((feature, value, delta_prob_pct))

        # 按影响绝对值排序，取前 10
        contributions_sorted = sorted(contributions, key=lambda x: abs(x[2]), reverse=True)
        explanations = []
        for feature, value, delta_prob in contributions_sorted[:10]:
            if delta_prob >= 0:
                explanations.append(f"{feature} = {value}，增大了 {abs(delta_prob):.1f}% 的风险（近似）")
            else:
                explanations.append(f"{feature} = {value}，减少了 {abs(delta_prob):.1f}% 的风险（近似）")
        return explanations, base_prob

    explanations, base_prob = shap_to_text_prob(explainer, phi, X_patient_series)

    # -------- 可加性自检 --------
    sum_logit = float(np.squeeze(explainer.expected_value)) + float(np.sum(phi))
    pred_prob_from_sum = sigmoid(sum_logit)  # 应≈ pred_prob（数值误差 ~1e-2 属正常）

    # -------- 组织输出 --------
    output = []
    output.append(f"背景基准概率（E[logit]→概率）: {base_prob:.2%}")
    output.append(f"该样本预测风险: {pred_prob:.2%}")
    output.append(f"加法性校验（sigmoid(E[logit] + Σφ) ≈ 预测概率）: {pred_prob_from_sum:.4f} vs {pred_prob:.4f}")
    output.append("按影响（概率百分点变化近似）排序的前 10 个特征：")
    output.extend(explanations)

    return output

# 1) 全部用文件路径
# out = explain_mods_kernel_shap(
#     df_patient="data.csv",
#     df_train="./Hs/mimichstrain-v1.csv",
#     scaler_file="./Hs/Hs_scaler.pkl",
#     model_file="./Hs/Hs_voting_model.pkl",
#     label_col=None,          # 若训练集最后一列为标签，可保持 None
#     patient_row=0,
#     background_size=100,
#     nsamples="auto"
# )
# print("\n".join(out))

# 2) 已在内存中的 DataFrame + 文件路径
# df_p = pd.read_csv("data.csv"); df_t = pd.read_csv("./Dead/mimicdeadtrain-v2.csv")
# out = explain_mods_kernel_shap(df_p, df_t, "./Dead/dead_scaler.pkl", "./Dead/dead_rf.pkl")
