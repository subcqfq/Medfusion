#Large Language Models and Traditional Machine Learning: Comparative Analysis and Model Merging for the Prediction of Four ICU Adverse Events from Structured Data

##Abstract

###Background and Objective:

Large language models (LLMs) show promise in medicine, but their performance on disease prediction from structured clinical data is unclear. This study compares LLMs with traditional machine learning (ML) algorithms for ICU outcome prediction and proposes a merged ML–LLM framework for clinical decision support.

###Methods:

Using the MIMIC-IV database, we selected 34 physiological variables to build four prediction tasks: mortality, hemorrhagic shock, hypoxemia, and multiple organ dysfunction syndrome (MODS). Seven ML models were compared with five mainstream LLMs using AUROC, AUPRC, and calibration. Furthermore, we designed a dual-layer architecture consisting of a “Tool Selection Agent” and a “Medical Expert Agent” to construct an ML–LLM merged model that performs intelligent tool routing and result integration.

###Results:

XGBoost outperformed all other ML models and all LLMs. For hemorrhagic shock, XGBoost achieved an AUROC of 0.829 [0.800–0.858] versus 0.673 [0.634–0.712] for the best LLM (Llama; p < 0.001), representing the largest performance gap. ML models showed better calibration and higher net clinical benefit across decision thresholds. Furthermore, the merged model successfully maintained high discriminative power while providing clinical rationale and actionable recommendations.

###Conclusions:

For prediction of four critical ICU adverse events from structured data, current LLMs cannot yet replace well-validated traditional ML algorithms. The proposed dual-layer merging architecture offers a practical route to more intelligent and interpretable clinical decision support.
