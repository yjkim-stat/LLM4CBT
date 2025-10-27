import os
import time
import json
import logging
from typing import Dict, List
from engine.space import Space
from openai import ChatCompletion
from string import Formatter

logger = logging.getLogger(__name__)


class Agent:
    def __init__(self, prompt_script: Dict) -> None:
        for k in ['api_inps', 'user']:
            assert k in prompt_script.keys(), f'{k} not in {prompt_script.keys()}'
        
        self.api_kwargs = prompt_script['api_inps']
        assert 'model' in self.api_kwargs.keys(), 'GPT Model should be specified in the api_inps'

        self.system_prompt = None
        if 'system' in prompt_script.keys():
            self.system_prompt = prompt_script['system']
            self.system_inputs = [i[1] for i in Formatter().parse(self.system_prompt)  if i[1] is not None]
        self.user_inputs = prompt_script['user']['inps']
        self.user_content = prompt_script['user']['content']

    def __repr__(self) -> str:
        line_div = os.getenv('LINE_DIV', '=')*100
        res = [line_div, '\tSystem : ', self.system_prompt, '\tUser : ', self.user_content, '\tInput : ', self.user_inputs, line_div]
        res = list(map(str, res))
        return '\n'.join(res)
    
    @staticmethod
    def fmt_prompt(prompt, var_space: Space):
        inputs = [i[1] for i in Formatter().parse(prompt)  if i[1] is not None]
        inputs = {k:v for k, v in var_space.values.items() if k in inputs}
        return prompt.format(**inputs) 
    
    def get_sys_prompt(self, var_space:Space):
        if self.system_prompt is not None:
            return self.fmt_prompt(self.system_prompt, var_space)
        else:
            return self.system_prompt
    
    def get_message(self, var_space: Space):
        inputs = {k:v for k, v in var_space.values.items() if k in self.user_inputs}

        try:
            return self.user_content.format(**inputs) 
        except KeyError as err:
            logger.error(f'WRONG space\n{self.user_content}\n{inputs}')
            raise KeyError(err)    
    
    def request(self, messages: List):
        start = time.time()
        response = ChatCompletion.create(messages=messages, **self.api_kwargs)
        latency = time.time() - start

        response_formated = response['choices'][0].message['content']
        if ('response_format' in self.api_kwargs):
            if self.api_kwargs['response_format']['type'] == 'json_object':
                response_formated = json.loads(response_formated)
        
        response_info = get_response_info(response, latency)

        return response_formated, response_info


def get_response_info(_response, _latency: float):
    res = dict()
    res['prompt_tokens'] = _response['usage']['prompt_tokens']
    res['completion_tokens'] = _response['usage']['completion_tokens']
    res['latency'] = f'{_latency:.2f}s'
    return res


class HumanAgent:
    def __init__(self, name='Human') -> None:
        self.name = name
        self.system_prompt = None
        self.user_inputs = []
        self.user_content = ""

    def get_sys_prompt(self, var_space: Space):
        return None

    def get_message(self, var_space: Space):
        return ""

    def request(self, messages: List):
        if messages:
            print("\n[Context from previous turn]")
            print(f"{messages[-1]['role'].capitalize()}: {messages[-1]['content']}")

        response = input("\n[Your Response] > ")
        return {'Response': response}, {"latency": "human"}
