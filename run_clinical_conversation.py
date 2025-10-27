"""Utilities to run patient-physician communication simulations.

This script relies on the existing `Field` orchestration engine to stage a
conversation between two LLM agents. The behaviour of each agent (patient and
physician) is defined through YAML prompt scripts located in the `prompts`
directory. Scenario specific context is provided through an external YAML
configuration file (see `configs/pancreatic_cancer_advanced.yml`).

Example usage
-------------

```bash
python run_clinical_simulation.py \
    --openai_api_key $OPENAI_API_KEY \
    --config configs/pancreatic_cancer_advanced.yml
```

The script will generate a markdown transcript and a tabular CSV file for each
scenario described in the configuration file.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from tqdm import tqdm
from pathlib import Path
from typing import Dict, Iterable, List

import openai
import pandas as pd
import yaml

from engine.field import Field
from engine.space import Space


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run clinical communication simulations")
    parser.add_argument("--openai_api_key", required=True, type=str, help="OpenAI API key")
    parser.add_argument(
        "--config",
        required=True,
        type=str,
        help="Path to a YAML configuration file describing the scenarios",
    )
    parser.add_argument(
        "--output_dir",
        default="./outputs/clinical",
        type=str,
        help="Directory where transcripts and metadata will be saved",
    )
    parser.add_argument(
        "--scenario_id",
        default=None,
        type=str,
        help="Optional scenario identifier to run a single scenario from the config",
    )
    parser.add_argument(
        "--turn_limit",
        default=None,
        type=int,
        help="Override the default number of generated turns defined in the config",
    )
    parser.add_argument(
        "--memory_turns",
        default=None,
        type=int,
        help="Override the number of previous turns shared with the agent prompts",
    )
    return parser.parse_args()


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _collect_space_variables(base_context: Dict[str, str], agent_inputs: Iterable[List[str]]) -> List[str]:
    """Collect the full set of variable names required by the prompts.

    Parameters
    ----------
    base_context:
        Scenario level context values that are always available.
    agent_inputs:
        An iterable containing the required input names for each agent prompt.

    Returns
    -------
    list
        A unique list of variable names to initialise the shared Space.
    """

    variables = set(base_context.keys())
    variables.update({"last_physician_message", "last_patient_message"})
    variables.update(
        {
            "patient_emotion",
            "patient_intent",
            "info_to_share",
            "clinical_intent",
            "next_step",
            "reassurance_point",
        }
    )
    for inputs in agent_inputs:
        variables.update(inputs)
    return sorted(variables)


def _normalise_objectives(values: Iterable[str]) -> str:
    return "; ".join(v.strip() for v in values if v is not None)


def _prepare_base_context(config: Dict, scenario: Dict) -> Dict[str, str]:
    patient_profile = scenario["patient_profile"]
    defaults = config.get("context_defaults", {})

    context = {
        "patient_name": str(patient_profile.get("name", "")),
        "patient_summary": str(patient_profile.get("summary", "")),
        "treatment_history": str(patient_profile.get("treatment_history", "")),
        "symptom_burden": str(patient_profile.get("symptom_burden", "")),
        "family_dynamics": str(scenario.get("family_dynamics", "")),
        "patient_emotion": str(patient_profile.get("emotional_state", "")),
        "conversation_focus": str(scenario.get("conversation_focus", "")),
        "patient_objectives": _normalise_objectives(scenario.get("patient_objectives", [])),
        "physician_objectives": _normalise_objectives(scenario.get("physician_objectives", [])),
        "care_setting": str(scenario.get("care_setting", defaults.get("care_setting", ""))),
        "time_constraints": str(
            scenario.get("time_constraints", defaults.get("time_constraints", ""))
        ),
        "physician_name": str(defaults.get("physician_name", "")),
        "institution": str(defaults.get("institution", "")),
    }

    cleaned_context = {k: v for k, v in context.items() if v is not None}
    return cleaned_context


def _load_response(response):
    if isinstance(response, dict):
        return response
    if isinstance(response, str):
        return json.loads(response)
    raise TypeError(f"Unexpected response type: {type(response)}")


def run_scenario(
    config: Dict,
    scenario: Dict,
    output_dir: Path,
    turn_limit: int,
    memory_turns: int,
) -> None:
    roles = config["roles"]
    patient_agent_name = roles.get("patient_agent_name", "Patient")
    physician_agent_name = roles.get("physician_agent_name", "Physician")

    field = Field()
    field.add_agent(patient_agent_name, roles["patient_prompt"])
    field.add_agent(physician_agent_name, roles["physician_prompt"])

    base_context = _prepare_base_context(config, scenario)

    space_vars = _collect_space_variables(base_context, field.get_agent_inputs())
    diagnosis_space = Space(scope=space_vars)
    diagnosis_space.sync(base_context)

    initial_message = scenario["initial_physician_message"].strip()
    field.add_chat(physician_agent_name, initial_message)
    diagnosis_space["last_physician_message"] = initial_message

    scenario_dir = ensure_directory(output_dir / scenario["id"])
    artifact_dir = ensure_directory(scenario_dir / "artifacts")
    artifact_index: List[Dict[str, str]] = []

    transcript_records: List[Dict] = []

    for turn_idx in tqdm(range(1, turn_limit + 1)):
        last_agent, last_utterance = field.get_last_chat()
        if last_agent == physician_agent_name:
            diagnosis_space["last_physician_message"] = last_utterance
            next_agent = patient_agent_name
        else:
            diagnosis_space["last_patient_message"] = last_utterance
            next_agent = physician_agent_name

        response, response_info, debug_payload = field.run(
            agent_name=next_agent,
            var_space=diagnosis_space,
            message_len=memory_turns,
            capture_debug=True,
        )
        response_payload = _load_response(response)

        if next_agent == patient_agent_name:
            utterance_key = "patient_utterance"
            diagnosis_space["last_patient_message"] = response_payload.get(utterance_key, "")
        else:
            utterance_key = "physician_utterance"
            diagnosis_space["last_physician_message"] = response_payload.get(utterance_key, "")

        utterance = response_payload.get(utterance_key)
        if utterance is None:
            raise KeyError(
                f"The response from {next_agent} did not contain the expected key '{utterance_key}'."
            )

        field.add_chat(next_agent, utterance)
        diagnosis_space.sync(response_payload)

        record = {
            "turn": turn_idx,
            "speaker": next_agent,
            "utterance": utterance,
            **{k: v for k, v in response_payload.items() if k != utterance_key},
            **response_info,
        }
        transcript_records.append(record)

        turn_key = f"turn_{turn_idx:02d}_{next_agent.lower()}"
        artifact_payload = {
            "turn": turn_idx,
            "speaker": next_agent,
            "utterance": utterance,
            "space_variables": debug_payload.get("space_values", {}),
            "system_prompt": debug_payload.get("system_prompt"),
            "user_prompt": debug_payload.get("user_prompt"),
            "request_messages": debug_payload.get("messages", []),
            "response_payload": response_payload,
            "response_info": response_info,
        }

        artifact_path = artifact_dir / f"{turn_key}.json"
        with artifact_path.open("w", encoding="utf-8") as fh:
            json.dump(artifact_payload, fh, ensure_ascii=False, indent=2)

        artifact_index.append(
            {
                "turn": turn_idx,
                "speaker": next_agent,
                "key": turn_key,
                "path": str(artifact_path.relative_to(scenario_dir)),
            }
        )

    transcript_path = scenario_dir / "transcript.md"
    with transcript_path.open("w", encoding="utf-8") as fh:
        fh.write(field.view_dialog())

    metadata_path = scenario_dir / "turns.csv"
    pd.DataFrame(transcript_records).to_csv(metadata_path, index=False)

    artifact_index_path = scenario_dir / "artifacts_index.json"
    with artifact_index_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "scenario_id": scenario["id"],
                "artifacts": artifact_index,
                "base_context": base_context,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    LOGGER.info("Scenario '%s' completed. Results stored in %s", scenario["id"], scenario_dir)


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()

    openai.api_key = args.openai_api_key

    config = load_config(args.config)

    turn_limit = args.turn_limit or config.get("turn_limit", 8)
    memory_turns = args.memory_turns or config.get("memory_turns", 4)

    scenarios = config.get("scenarios", [])
    if args.scenario_id:
        scenarios = [s for s in scenarios if s.get("id") == args.scenario_id]
        if not scenarios:
            raise ValueError(f"Scenario '{args.scenario_id}' was not found in the configuration file")

    output_dir = ensure_directory(args.output_dir)

    for scenario in scenarios:
        LOGGER.info(f"Running {scenario}")
        
        run_scenario(config, scenario, output_dir, turn_limit, memory_turns)


if __name__ == "__main__":
    main()

