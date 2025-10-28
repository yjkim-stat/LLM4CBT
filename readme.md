# LLM4CBT: Aligning Large Language Models for Cognitive Behavioral Therapy

**Official Reference Implementation** for  
📄 [*Kim et al., "Aligning Large Language Models for Cognitive Behavioral Therapy: A Proof-of-Concept Study,"* Frontiers in Psychiatry (2025)](https://doi.org/10.3389/fpsyt.2025.1583739)

[![Paper](https://img.shields.io/badge/Paper-PDF-red)](https://doi.org/10.3389/fpsyt.2025.1583739)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
---

## 🧠 Overview

**LLM4CBT** provides a simulation environment for evaluating and aligning large language models (LLMs) to principles of **Cognitive Behavioral Therapy (CBT)**.  
The framework enables controlled, reproducible **clinical communication simulations** between virtual patients and AI-based therapists.

---

## ⚙️ Installation

To set up the environment:

```bash
conda create -n llm4cbt python=3.10
conda activate llm4cbt
pip install openai==0.28 pandas numpy transformers torch sentencepiece accelerate
````

---

## 💬 Clinical Communication Simulation

Run the main clinical conversation simulation with:

```bash
python run_clinical_conversation.py \
    --openai_api_key $OPENAI_API_KEY \
    --config configs/pancreatic_cancer_advanced.yml \
    --output_dir ./outputs/clinical
```

### Optional Arguments

| Argument         | Description                                                 |
| ---------------- | ----------------------------------------------------------- |
| `--scenario_id`  | Run a single scenario by ID instead of all configured cases |
| `--turn_limit`   | Override the default number of conversation turns           |
| `--memory_turns` | Override memory window size for contextual recall           |

---

## 📁 Output Structure

Simulation outputs are organized by scenario under the specified output directory.

| File                               | Description                                                                    |
| ---------------------------------- | ------------------------------------------------------------------------------ |
| `transcript.md`                    | Full conversation transcript in turn-by-turn order                             |
| `turns.csv`                        | Detailed metadata for each conversational turn                                 |
| `artifacts/turn_XX_<speaker>.json` | Per-turn artifacts containing API inputs, variable states, and raw completions |
| `artifacts_index.json`             | Index summarizing artifacts and base context used during simulation            |

---

## 📚 Citation

If you use **LLM4CBT** in your research, please cite both the paper and this implementation repository.

### 📝 Original Paper

```bibtex
@article{kim2025aligning,
  title={Aligning large language models for cognitive behavioral therapy: a proof-of-concept study},
  author={Kim, Yejin and Choi, Chi-Hyun and Cho, Selin and Sohn, Jy-yong and Kim, Byung-Hoon},
  journal={Frontiers in Psychiatry},
  volume={16},
  pages={1583739},
  year={2025},
  publisher={Frontiers}
}
```

### 💻 Implementation Repository

```bibtex
@software{kim2025llm4cbt,
  author       = {Kim, Yejin and Choi, Chi-Hyun and Cho, Selin and Sohn, Jy-yong and Kim, Byung-Hoon},
  title        = {LLM4CBT: Reference Implementation for "Aligning Large Language Models for Cognitive Behavioral Therapy"},
  year         = {2025},
  version      = {v1.0.0},
  publisher    = {Yonsei ITML Lab},
  url          = {https://github.com/Yonsei-ITML/LLM4CBT},
  note         = {GitHub repository implementing the methods described in Kim et al. (2025), *Frontiers in Psychiatry*}
}
```

---

## 🧩 Acknowledgments

This work was conducted by the **Intelligent Technology and Machine Learning (ITML) Lab**,
**Yonsei University**, Seoul, Korea.
