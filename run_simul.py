import os
import json
import random
import logging
import argparse
import pandas as pd
from pprint import pformat
from itertools import chain

import openai

import Persona
from src.logger import Logger
from engine.space import Space
from engine.field import Field
from utils import name_map

logging.basicConfig(level=logging.INFO)


parser = argparse.ArgumentParser()

# GPT
parser.add_argument('--openai_api_key', type=str, required=False)

# Execution option
parser.add_argument('--seed', type=str, default=1234)
parser.add_argument('--scenario', type=str, default='common')
parser.add_argument('--sample_idx', type=str, default='depression_persona')
parser.add_argument('--turn_limit', type=int, default=5)

parser.add_argument('--prompt_therapist', type=str, default=None)
parser.add_argument('--prompt_client', type=str, required=True)

parser.add_argument('--emotion', type=str) # 감정 추가

def generate_client_behavior(_scenario, _cnt, _q1, _q2, _q3):
    if _scenario == 'common':
        if _cnt > _q3:
            _aha_moment = 'Yes'  
            _behavior = 'changing unhealty behavior'
        elif _cnt > _q2 :
            _aha_moment = 'No'  
            _behavior = 'gained insight'
        elif _cnt > _q1 :
            _aha_moment = 'No'  
            _behavior = 'sustaining unhealthy behavior'
        else:
            _aha_moment = 'No'  
            _behavior = random.choice(['sharing emotions', 'sharing experiences'])
    elif _scenario == 'simul':
        if _cnt > _q3:
            _aha_moment = 'Yes'  
            _behavior = 'sharing how you perceive the experience now'
        elif _cnt > _q2 :
            _aha_moment = 'No'  
            _behavior = "sharing reaction during the experience"
        elif _cnt > _q1 :
            _aha_moment = 'No'  
            _behavior = 'explain experiences'
        else:
            _aha_moment = 'No'  
            _behavior = 'sharing emotions'
    elif _scenario == 'resistance':
        _aha_moment = 'No'  
        _behavior = 'feel reluctant to talk about your story'
    elif _scenario == 'overwhelmed':
        _aha_moment = 'No'  
        _behavior = 'feel overwhelmed to the recent experiences and symptoms'
    elif _scenario == 'atl':
        if _cnt > _q3:
            _aha_moment = 'Yes'  
            _behavior = 'sharing schema or key belief'
        elif _cnt > _q2 :
            _aha_moment = 'No'  
            _behavior = 'sharing concise statement'
        elif _cnt > _q1 :
            _aha_moment = 'No'  
            _behavior = 'sharing complicate thoughts'
        else:
            _aha_moment = 'No'  
            _behavior = random.choice(['sharing emotions', 'sharing experiences'])        
    elif _scenario == 'defector':
        return random.choice(['sharing emotions', 'sharing experiences'], p=[0.3, 0.7])
    else:
        raise ValueError(f'Cannot support scenario : {_scenario}')
    return _behavior, _aha_moment


if __name__ == '__main__':
    TURN_LIMIT = 9
    symptom = 'GAD'
    version = 3
    SCENARIO = 'simul'
    API_KEY = 'TODO'
    sample_idx = f'patient-{symptom}-v{version}'
    args = parser.parse_args(args=[
        '--openai_api_key', API_KEY,
        '--scenario', SCENARIO,
        '--prompt_client', 'patient',
        '--prompt_therapist', 'human',
        '--sample_idx', sample_idx,
        '--turn_limit', str(TURN_LIMIT)
    ])

    assert args.openai_api_key != 'TODO', "OpenAI의 API key를 입력해주세요!"
    output_dir = f'./outputs/simul/{args.sample_idx}-{args.scenario}-{args.prompt_client}-{args.seed}'
    run_name = []
    if args.prompt_therapist is not None:
        run_name.append(args.prompt_therapist)
    else:
        run_name.append('MultiTherapist')
    run_name = '-'.join(run_name)
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(f'{output_dir}/{run_name}', exist_ok=True)

    logger = Logger(run_name, output_dir)
    logger.addFileHandler(f'{run_name}-log.txt')

    for k, v in vars(args).items():
        if v == args.openai_api_key:
            continue
        logger.info(f'\n\t{k} : {v}')        

    openai.api_key = args.openai_api_key

    c_symptom = Persona.story_dict[args.sample_idx]['Input']['SYMPTOM']
    c_description= Persona.story_dict[args.sample_idx]['Input']['DESCRIPTION']
    c_situation = Persona.story_dict[args.sample_idx]['Input']['SITUATION']
    c_reaction = Persona.story_dict[args.sample_idx]['Input']['REACTION']
    c_automatic_thought = Persona.story_dict[args.sample_idx]['Input']['AUTOMATIC_THOUGHT']

    field = Field()
    field.add_agent(name_map[args.prompt_client], args.prompt_client) 

    agent_key = {
        'therapist-base': 'Therapist Naive',
        'therapist-downarrow': 'Therapist DownArrow',
        'human':' Therapist Human',
        }  
    
    if args.prompt_therapist == 'human':
        field.add_agent(agent_key[args.prompt_therapist], prompt_fname=args.prompt_therapist, human=True)
    else:
        field.add_agent(agent_key[args.prompt_therapist], prompt_fname=args.prompt_therapist)

    space_vars = [['automatic_thoughts'],['client_symptom'],['description'],['client_situation'], ['c_reaction']] + field.get_agent_inputs()
    space_vars = list(chain.from_iterable(space_vars))
    assert isinstance(space_vars, list)
    
    diagnosis_space = Space(scope=space_vars)

    diagnosis_space['client_symptom'] = c_symptom
    diagnosis_space['description'] = c_description
    diagnosis_space['client_situation'] = c_situation
    diagnosis_space['client_reaction'] = c_reaction
    
    diagnosis_space['automatic_thoughts'] = c_automatic_thought
    diagnosis_space['client_mood'] = args.emotion

    field.add_chat(agent_name='Therapist', utterance='hi, nice to see you today how you been going?')

    response_tab = []
    counts = 1
    keep_therapy = True
    while keep_therapy:
        last_agent, last_utterance = field.get_last_chat()
        if last_agent == 'Client':
            logger.info(f'{counts} th step, Therapist turn')

            if args.prompt_therapist is not None:
                current_agent = field.get_agent_with_key(args.prompt_therapist)
            else:
                router_response_formated, response_info = field.run(
                    agent_name='Router', 
                    var_space=diagnosis_space, 
                    message_len=args.turn_limit)
                current_agent = router_response_formated['Therapist']
                current_agent = field.get_agent_with_key(current_agent)

            diagnosis_space.sync(dict(
                user_utterance=last_utterance,
                selected_therapist = current_agent
            ))

        elif 'Therapist' in last_agent:
            logger.info(f'{counts} th step, Client turn')
            current_agent = 'Client'
            q1, q2, q3 = args.turn_limit * 1/4 , args.turn_limit * 2/4, args.turn_limit * 3/4
            behavior, aha_moment = generate_client_behavior(args.scenario, counts, q1, q2, q3)
            
            diagnosis_space.sync(dict(
                aha_moment=aha_moment,
                conversational_behavior_gt=behavior,
                therapist_utterance=last_utterance
                ))
        else:
            raise KeyError(f'Wrong Agent')
        
        response_formated, response_info = field.run(
            agent_name=current_agent, 
            var_space=diagnosis_space, 
            message_len=args.turn_limit+1)
        
        if current_agent == 'Client':
            if isinstance(response_formated, str):
                try:
                    current_utterance = json.loads(response_formated)['Client_Response']
                except json.decoder.JSONDecodeError as err:
                    logger.info(f'CANNOT DECODE :\n{response_formated}')
                    raise json.decoder.JSONDecodeError(err)
            elif isinstance(response_formated, dict):
                current_utterance = response_formated['Client_Response']
            else:
                print(f'{type(response_formated)}\n{response_formated}')
                raise TypeError()
        elif 'Therapist' in current_agent:
            current_utterance = response_formated['Response']
        else:
            raise KeyError(f'Wrong Agent')
        field.add_chat(current_agent, current_utterance)

        if isinstance(response_formated, dict):
            diagnosis_space.sync(response_formated)
        
        agent_name, agent_utterance = field.get_last_chat()
        response_tab.append(dict({'role': agent_name, 'content': agent_utterance}, **diagnosis_space.values))

        logger.info(f'{counts} DONE')
        counts += 1
        if counts >= args.turn_limit:
            keep_therapy = False   

    response_tab = pd.DataFrame(response_tab)
    response_tab.to_csv(f'{output_dir}/{run_name}/experiments.csv')

    with open(f'{output_dir}/{run_name}/dialog.md', 'w', encoding='utf-8') as f:
        f.write(field.view_dialog())

