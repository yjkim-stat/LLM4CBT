import os
import logging
import pandas as pd

from copy import deepcopy

import prompts
from engine.space import Space
from engine.agent import Agent, HumanAgent

logger = logging.getLogger(__name__)
logger.addHandler(logging.FileHandler(f'./logs/{__name__}.log', 'w', 'utf-8'))

class Field:
    PRIMARY_KEY_FORMAT = 'STEP:{time}'

    def __init__(self) -> None:
        self.agents = dict()
        self.dialog = pd.DataFrame()
        self.key_agents = []

    def add_agent(self, agent_name, prompt_fname=None, shared_llama=None, human=False):
        if human:            
            new = HumanAgent(name=agent_name)
        else:
            prompt_script = prompts.prompt_dict[prompt_fname]
            self.key_agents.append((prompt_fname, agent_name))
            if prompt_script['api_inps'].get('provider', 'openai') == 'llama':
                raise NotImplementedError('이 버전은 Llama를 지원하지 않습니다.')
            else:
                new = Agent(prompt_script=prompt_script)

        self.key_agents.append((prompt_fname, agent_name))
        self.agents[agent_name] = new

    def get_agent_with_key(self, prompt_fname):
        key_to_agent_name = {x[0]:x[1] for x in self.key_agents}
        return key_to_agent_name[prompt_fname]

    def delete_agent(self):
        raise NotImplementedError()

    def search_last_index_time(self):
        indices = self.dialog.index
        if len(indices) == 0:
            time = 0
        else:
            indices_times = list(map(lambda x: int(x.split(':')[-1]), indices))
            time = max(indices_times)
        return time
    
    def add_chat(self, agent_name, utterance):
        time = self.search_last_index_time() + 1
        primary_key = Field.PRIMARY_KEY_FORMAT.format(time=time)
        self.dialog.loc[primary_key, agent_name] = utterance

        # Sanity check
        indices_times = list(map(lambda x: int(x.split(':')[-1]), self.dialog.index))
        for sorted_idx, org_idx in zip(sorted(indices_times), indices_times):
            assert sorted_idx == org_idx

    def delete_chat(self):
        raise NotImplementedError()

    def get_chat(self, primary_key):
        assert primary_key in self.dialog.index, f'{primary_key} not in {self.dialog.index}'
        chat = self.dialog.loc[primary_key].dropna()

        agent_name = chat.index[0]
        agent_utterance = chat[agent_name]
        return agent_name, agent_utterance
    
    def get_last_chat(self):
        time = self.search_last_index_time()
        primary_key = Field.PRIMARY_KEY_FORMAT.format(time=time)
        return self.get_chat(primary_key)
    
    def get_agent_inputs(self):
        res = list(map(lambda agent: agent.user_inputs, self.agents.values()))
        return res

    def view_agents(self):
        """
        View detailed information of enrolled agents
        """
        line_div = os.getenv('LINE_DIV', '=')*100
        res = []
        for agent_name, agent_desc in zip(self.agents.keys(), list(map(str, self.agents.values()))):
            res.extend([line_div, agent_name, agent_desc, line_div])
        return '\n'.join(res)

    def view_dialog(self):
        res = []
        for idx in self.dialog.index:
            agent_name, agent_utterance = self.get_chat(idx)
            res.append(f'{agent_name}:{agent_utterance}')
            if 'Therapist' in agent_name:
                res.append('\n')
        return '\n'.join(res)
    
    def run(self, agent_name, var_space: Space, message_len:int, capture_debug: bool = False):
        agent = self.agents[agent_name]

        msgs = []
        prompt_sys=None
        if agent.system_prompt is not None:
            prompt_sys = agent.get_sys_prompt(var_space)
            update_msg(msgs, 'system', prompt_sys)

        prompt_usr = agent.get_message(var_space)

        if message_len > 0:
            previous_chats = self.dialog.iloc[-message_len:]
            for line_idx in range(len(previous_chats)):
                line = previous_chats.iloc[line_idx]
                
                if (agent_name in previous_chats.columns) and (str(line[agent_name]).lower() != 'nan'):
                    role = 'user'
                    chat = line[agent_name]
                else:
                    role = 'assistant'
                    chat = line.dropna().item()

                update_msg(msgs, role, chat)

        if capture_debug:
            space_snapshot = deepcopy(var_space.values)
        else:
            space_snapshot = None

        if not isinstance(agent, HumanAgent):
            update_msg(msgs, 'user', prompt_usr)

        if capture_debug:
            messages_snapshot = deepcopy(msgs)
        else:
            messages_snapshot = None

        response_json, response_info = agent.request(msgs)

        if capture_debug:
            debug_payload = {
                'space_values': space_snapshot,
                'messages': messages_snapshot if messages_snapshot is not None else deepcopy(msgs),
                'system_prompt': prompt_sys,
                'user_prompt': prompt_usr,
            }
            return response_json, response_info, debug_payload

        return response_json, response_info

def update_msg(_msg, _role, _content):
    _msg.append({'role': _role, 'content':_content})        
