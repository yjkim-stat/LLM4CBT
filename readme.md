# Installation

```
conda create -n llm4cbt python=3.10
conda activate llm4cbt
pip install openai==0.28 pandas numpy transformers torch sentencepiece accelerate
```

## Clinical communication simulation

```bash
python run_clinical_conversation.py \
    --openai_api_key $OPENAI_API_KEY \
    --config configs/pancreatic_cancer_advanced.yml \
    --output_dir ./outputs/clinical
```

* `--scenario_id`를 지정하면 단일 시나리오만 실행할 수 있습니다.
* `--turn_limit` 및 `--memory_turns` 옵션으로 기본 설정을 덮어쓸 수 있습니다.

시뮬레이션 결과는 각 시나리오별 디렉터리에 저장되며, 다음과 같은 산출물을 제공합니다.

* `transcript.md` – 턴 순서에 따른 대화 로그
* `turns.csv` – 각 턴의 세부 메타데이터 테이블
* `artifacts/turn_XX_<speaker>.json` – 해당 턴을 생성할 때 사용된 변수 스냅샷, API 요청 메시지, 완전한 출력 정보를 포함한 개별 아티팩트
* `artifacts_index.json` – 아티팩트 파일과 기본 컨텍스트를 요약한 인덱스

---

# How to Cite

## Original Paper

```
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

## Implementation Repository
```
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
