
SYMPTOM=GAD
VERSION=v1

SAMPLE_IDX=patient-${SYMPTOM}-${VERSION}
python run_simul.py --openai_api_key TODO \
    --scenario $SCENARIO \
    --prompt_client patient\
    --prompt_therapist therapist-downarrow\
    --sample_idx $SAMPLE_IDX\
    --turn_limit $TURN_LIMIT\