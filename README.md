🏛️ Decoding Decentralized Governance: Fine-Tuning LLMs for DAO Analysis

Fine-tuning GPT-4o to analyze organizational innovation in Decentralized Autonomous Organizations (DAOs)

📋 Overview

DAOs represent a radical shift from hierarchical to community-driven governance. This research fine-tunes GPT-4o to act as a specialized "Innovation Auditor" for Web3 organizations, capable of analyzing governance proposals and categorizing innovations at scale.

Problem: Traditional analysis methods are slow and biased. General LLMs lack DAO-specific expertise.

Solution: Domain-specific fine-tuning on 250 real DAO governance proposals.

📊 Dataset
250 proposals from 5 major DAOs (Uniswap, Aave, MakerDAO, Optimism, Arbitrum)
19 proposal types covering governance, protocol upgrades, grants, treasury, security
Time period: December 2024 - December 2025
Format: JSONL for OpenAI fine-tuning

🤗 Dataset: arad1367/DAOs

🤖 Models
Model	ID	Purpose
Base	gpt-4o-2024-08-06	General-purpose LLM
Fine-Tuned	ft:gpt-4o-2024-08-06:personal:dao-16:Cqbyeyt6	DAO governance specialist

🔧 Training
Tokens: 68,287
Epochs: 1
Batch Size: 16
Learning Rate Multiplier: 2
📈 Evaluation

11 metrics across technical and qualitative dimensions:

Technical: Cosine Similarity, ROUGE-L, Perplexity, Word Count, Token Usage, BERTScore, METEOR, BLEU

Qualitative (LLM-as-a-Judge): Innovation Relevance, Governance Clarity, Technical Accuracy

🚀 Usage
pip install openai pandas matplotlib seaborn scikit-learn rouge-score bert-score nltk
python evaluate_optimized.py

📧 Contact

Pejman Ebrahimi
📧 pejman.ebrahimi77@gmail.com

📝 Citation
@article{dao_governance_llm_2025,
  title={Decoding Decentralized Governance: Fine-Tuning LLMs for DAO Analysis},
  author={Ebrahimi, Pejman},
  year={2025},
  note={https://huggingface.co/datasets/arad1367/DAOs}
}

🔗 Links
📊 Dataset
🎯 Evaluation Script

⭐ Star this repo if you find it useful! ⭐
