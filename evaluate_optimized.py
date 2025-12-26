import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import numpy as np
import math
import json
from bert_score import score as bert_score
from nltk.translate.meteor_score import meteor_score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import time

# --- CONFIGURATION ---
OPENAI_API_KEY = "YOUR OPENAI KEY" 

# Models
JUDGE_MODEL = "gpt-4.1-mini-2025-04-14"
BASE_MODEL = "gpt-4o-2024-08-06"
FT_MODEL = "ft:gpt-4o-2024-08-06:personal:dao-16:Cqbyeyt6"

# --- TEST DATA ---
test_data = [
    {
        "question": "Analyze the impact of MakerDAO's parameter change proposal on stability fees.",
        "reference": "The proposal to adjust MakerDAO's stability fee aims to better align borrowing costs with market conditions, potentially stabilizing the DAI peg and incentivizing appropriate borrowing behavior."
    },
    {
        "question": "Evaluate the effectiveness of DAO governance in the Aragon Network's dispute resolution mechanism.",
        "reference": "Aragon's dispute resolution leverages decentralized juries to ensure fair outcomes, enhancing trust and participation, though scalability and jury selection remain challenges."
    },
    {
        "question": "What are the innovation implications of MolochDAO's grant funding model?",
        "reference": "MolochDAO's grant funding encourages community-driven innovation by pooling resources and enabling transparent, collective decision-making, fostering incremental and radical innovations."
    },
    {
        "question": "How does the proposal for quadratic voting in DAO governance affect decision-making quality?",
        "reference": "Quadratic voting reduces the influence of large stakeholders, promoting more democratic and nuanced decision-making, which can lead to higher quality governance outcomes."
    },
    {
        "question": "Assess the risks and benefits of DAO treasury diversification strategies.",
        "reference": "Diversifying DAO treasuries mitigates financial risks and exposure to single asset volatility, but requires careful governance to balance risk and opportunity costs."
    },
    {
        "question": "Analyze the role of token-weighted voting in shaping DAO innovation trajectories.",
        "reference": "Token-weighted voting aligns incentives with investment but may centralize power, potentially stifling diverse innovation perspectives within the DAO."
    },
    {
        "question": "Evaluate the proposal to implement off-chain governance signaling in DAOs.",
        "reference": "Off-chain signaling can increase participation and reduce on-chain costs, but may lack enforceability and transparency, impacting governance effectiveness."
    },
    {
        "question": "What impact does the introduction of reputation systems have on DAO member engagement?",
        "reference": "Reputation systems incentivize active participation and accountability, enhancing community cohesion and innovation, though they must be designed to prevent gaming."
    },
    {
        "question": "How do DAO governance proposals address scalability challenges in decentralized decision-making?",
        "reference": "Proposals often include delegation, layered governance, or off-chain mechanisms to improve scalability while maintaining decentralization and inclusivity."
    },
    {
        "question": "Assess the potential of AI integration in automating DAO proposal evaluations.",
        "reference": "AI can streamline proposal analysis by providing rapid, unbiased assessments, improving decision speed and quality, but requires careful tuning to DAO-specific contexts."
    },
    {
        "question": "Analyze the implications of introducing time-locked voting in DAO governance.",
        "reference": "Time-locked voting can encourage long-term commitment and reduce short-term manipulation, fostering stability in governance decisions."
    },
    {
        "question": "Evaluate the effectiveness of multisig wallets in DAO treasury management.",
        "reference": "Multisig wallets enhance security by requiring multiple approvals for transactions, reducing risks of misappropriation but may slow down urgent decisions."
    },
    {
        "question": "What are the challenges of implementing cross-DAO collaboration proposals?",
        "reference": "Cross-DAO collaboration faces challenges in aligning incentives, governance models, and technical interoperability, but can unlock synergies and shared innovation."
    },
    {
        "question": "How does the use of prediction markets influence DAO decision-making?",
        "reference": "Prediction markets can aggregate diverse insights and improve forecasting accuracy, aiding better governance decisions, though they require careful design to avoid manipulation."
    },
    {
        "question": "Assess the role of community forums in shaping DAO proposal quality.",
        "reference": "Community forums facilitate discussion and feedback, improving proposal quality and inclusiveness, but require moderation to maintain constructive dialogue."
    },
    {
        "question": "Analyze the impact of gas fee optimization proposals on DAO participation.",
        "reference": "Reducing gas fees lowers barriers to participation, increasing voter turnout and inclusivity, but may require trade-offs in security or decentralization."
    },
    {
        "question": "Evaluate the proposal for integrating off-chain identity verification in DAOs.",
        "reference": "Off-chain identity verification can enhance trust and reduce Sybil attacks, but raises privacy concerns and may conflict with decentralization principles."
    },
    {
        "question": "What are the innovation outcomes of implementing modular governance frameworks in DAOs?",
        "reference": "Modular governance allows flexible adaptation and experimentation, fostering innovation by enabling DAOs to customize governance components."
    },
    {
        "question": "How do DAO proposals address the challenge of voter apathy?",
        "reference": "Proposals include incentives, delegation, and simplified voting mechanisms to increase engagement and reduce voter apathy."
    },
    {
        "question": "Assess the potential benefits and risks of automated smart contract upgrades in DAOs.",
        "reference": "Automated upgrades can accelerate innovation and patch vulnerabilities but risk unintended consequences if not properly governed."
    }
]

# --- INITIALIZATION ---
client = OpenAI(api_key=OPENAI_API_KEY)
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
smooth_fn = SmoothingFunction().method4

# --- HELPER FUNCTIONS ---
def get_embedding(text):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding

def calculate_perplexity(logprobs):
    if not logprobs: return 0
    values = [token.logprob for token in logprobs]
    avg_logprob = sum(values) / len(values)
    return math.exp(-avg_logprob)

def get_completion_data(model, prompt):
    try:
        start_time = time.time()
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            logprobs=True,
            top_logprobs=1
        )
        latency = time.time() - start_time
        content = response.choices[0].message.content
        ppl = calculate_perplexity(response.choices[0].logprobs.content)
        word_count = len(content.split())
        token_count = response.usage.total_tokens if hasattr(response, 'usage') else word_count * 1.3  # approximate
        return content, ppl, word_count, token_count, latency
    except Exception as e:
        print(f"Error generating completion: {e}")
        return "", 0, 0, 0, 0

def compute_bert_score(cands, refs):
    P, R, F1 = bert_score(cands, refs, lang='en', rescale_with_baseline=True)
    return float(F1.mean())

def compute_meteor(ref, hyp):
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    return meteor_score([ref_tokens], hyp_tokens)

def compute_bleu(ref, hyp):
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    return sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smooth_fn)

# --- LLM-AS-A-JUDGE FUNCTION ---
def evaluate_qualitative_metrics(question, answer, reference):
    system_prompt = """You are an expert evaluator of DAO governance and innovation. 
    Evaluate the AI's response to a question about DAO governance based on the following criteria.
    Return the result as a valid JSON object with keys: 'Innovation_Relevance_Score', 'Governance_Clarity_Score', 'Technical_Accuracy_Score', 'Reasoning'.

    Criteria:
    1. Innovation_Relevance_Score (1-5): 
       - 1: Completely misses the innovation aspect or provides irrelevant information.
       - 5: Clearly identifies and analyzes the innovation implications, distinguishing between incremental and radical innovation.
    2. Governance_Clarity_Score (1-5):
       - 1: Confusing explanation of governance mechanisms or lacks key details.
       - 5: Provides a clear, comprehensive explanation of governance structures and processes.
    3. Technical_Accuracy_Score (1-5):
       - 1: Contains factual errors or misunderstandings about blockchain/Web3 concepts.
       - 5: Technically accurate with correct terminology and conceptual understanding.
    """

    user_prompt = f"""
    Question: {question}
    Reference Answer: {reference}
    AI Response to Evaluate: {answer}

    Provide the JSON evaluation.
    """

    try:
        response = client.chat.completions.create(
            model=JUDGE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        print(f"Error in judging: {e}")
        return {"Innovation_Relevance_Score": 0, "Governance_Clarity_Score": 0, "Technical_Accuracy_Score": 0, "Reasoning": "Error"}

# --- MAIN EVALUATION LOOP ---
results = []
print(f"Starting Evaluation on {len(test_data)} examples...")

for i, item in enumerate(test_data): 
    print(f"Processing {i+1}/{len(test_data)}...")
    q = item['question']
    ref = item['reference']
    ref_emb = get_embedding(ref)

    for model_name, model_id in [("Base", BASE_MODEL), ("Fine-Tuned", FT_MODEL)]:
        ans, ppl, wc, token_count, latency = get_completion_data(model_id, q)
        emb = get_embedding(ans)
        sim = cosine_similarity([ref_emb], [emb])[0][0]
        rouge = scorer.score(ref, ans)['rougeL'].fmeasure
        bert = compute_bert_score([ans], [ref])
        meteor = compute_meteor(ref, ans)
        bleu = compute_bleu(ref, ans)
        qual_metrics = evaluate_qualitative_metrics(q, ans, ref)

        results.append({"Model": model_name, "Metric": "Semantic Similarity", "Score": sim})
        results.append({"Model": model_name, "Metric": "ROUGE-L", "Score": rouge})
        results.append({"Model": model_name, "Metric": "Perplexity", "Score": ppl})
        results.append({"Model": model_name, "Metric": "Word Count", "Score": wc})
        results.append({"Model": model_name, "Metric": "Token Count", "Score": token_count})
        results.append({"Model": model_name, "Metric": "Latency (s)", "Score": latency})
        results.append({"Model": model_name, "Metric": "BERTScore", "Score": bert})
        results.append({"Model": model_name, "Metric": "METEOR", "Score": meteor})
        results.append({"Model": model_name, "Metric": "BLEU", "Score": bleu})

        results.append({"Model": model_name, "Metric": "Innovation Relevance (1-5)", "Score": qual_metrics['Innovation_Relevance_Score']})
        results.append({"Model": model_name, "Metric": "Governance Clarity (1-5)", "Score": qual_metrics['Governance_Clarity_Score']})
        results.append({"Model": model_name, "Metric": "Technical Accuracy (1-5)", "Score": qual_metrics['Technical_Accuracy_Score']})

# --- VISUALIZATION ---
df = pd.DataFrame(results)
sns.set_theme(style="whitegrid")

# Technical metrics to plot (excluding latency for clarity)
technical_metrics = ["Semantic Similarity", "ROUGE-L", "Perplexity", "Word Count", "Token Count", "BERTScore", "METEOR", "BLEU"]
titles_tech = [
    "Factual Accuracy (Cosine Similarity)", 
    "Structural Alignment (ROUGE-L)", 
    "Model Uncertainty (Perplexity)", 
    "Verbosity (Word Count)",
    "Token Usage",
    "Semantic Similarity (BERTScore)",
    "Lexical & Semantic Match (METEOR)",
    "Fluency & Adequacy (BLEU)"
]

fig1, axes1 = plt.subplots(3, 3, figsize=(18, 15))
axes1 = axes1.flatten()

for i, metric in enumerate(technical_metrics):
    subset = df[df['Metric'] == metric]
    sns.boxplot(x="Model", y="Score", data=subset, ax=axes1[i], palette=["#95a5a6", "#3498db"], width=0.5)

    means = subset.groupby("Model")["Score"].mean()
    for j, model in enumerate(["Base", "Fine-Tuned"]):
        if model in means:
            val = means[model]
            axes1[i].text(j, val, f'{val:.3f}', ha='center', va='bottom', fontweight='bold', 
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    axes1[i].set_title(titles_tech[i], fontsize=14)
    axes1[i].set_xlabel("")

# Hide unused subplot (9th)
axes1[-1].axis('off')

plt.tight_layout()
fig1.savefig("Figure_1_Technical_Metrics_Enhanced.png", dpi=300)
print("Saved 'Figure_1_Technical_Metrics_Enhanced.png'")

# Qualitative metrics plot (same as before)
fig2, axes2 = plt.subplots(1, 3, figsize=(16, 6))
qualitative_metrics = ["Innovation Relevance (1-5)", "Governance Clarity (1-5)", "Technical Accuracy (1-5)"]
titles_qual = [
    "Innovation Relevance (Judge Score)", 
    "Governance Clarity (Judge Score)", 
    "Technical Accuracy (Judge Score)"
]

for i, metric in enumerate(qualitative_metrics):
    subset = df[df['Metric'] == metric]

    sns.boxplot(x="Model", y="Score", data=subset, ax=axes2[i], palette=["#e74c3c", "#2ecc71"], width=0.5, showfliers=False)
    sns.stripplot(x="Model", y="Score", data=subset, ax=axes2[i], color="black", alpha=0.3, jitter=True)

    means = subset.groupby("Model")["Score"].mean()
    for j, model in enumerate(["Base", "Fine-Tuned"]):
        if model in means:
            val = means[model]
            axes2[i].text(j, val, f'{val:.2f}', ha='center', va='bottom', fontweight='bold', 
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    axes2[i].set_title(titles_qual[i], fontsize=14)
    axes2[i].set_xlabel("")
    axes2[i].set_ylim(1, 5.5)

plt.tight_layout()
fig2.savefig("Figure_2_Qualitative_Metrics.png", dpi=300)
print("Saved 'Figure_2_Qualitative_Metrics.png'")

# --- FINAL SUMMARY TABLE ---
summary = df.groupby(["Model", "Metric"])["Score"].mean().unstack()
print("\n--- Final Results for Paper ---")
print(summary)