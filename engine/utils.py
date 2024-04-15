import os
import openai
import numpy as np

from .step_interpreters import register_step_interpreters, parse_step


class Program:
    def __init__(self, prog_str, init_state=None):
        self.prog_str = prog_str
        self.state = init_state if init_state is not None else dict()
        self.instructions = self.prog_str.split('\n')


class ProgramInterpreter:
    def __init__(self, dataset='MEVQA'):
        self.step_interpreters = register_step_interpreters(dataset)

    def execute_step(self, prog_step, inspect):
        step_name = parse_step(prog_step.prog_str, partial=True)['step_name']
        print(step_name)
        return self.step_interpreters[step_name].execute(prog_step, inspect)

    def execute(self, prog, init_state, inspect=False):
        if isinstance(prog, str):
            prog = Program(prog, init_state)
        else:
            assert (isinstance(prog, Program))

        prog_steps = [Program(instruction, init_state=prog.state) \
                      for instruction in prog.instructions]

        pro_str = '<hr>' if isinstance(inspect, str) and inspect.casefold() == 'html' else ''
        for prog_step in prog_steps:
            if isinstance(inspect, str) and inspect.casefold() == 'html':
                step_output, step_html = self.execute_step(prog_step, inspect)
                pro_str += step_html + '<hr>'
            elif isinstance(inspect, str) and inspect.casefold() == 'text':
                step_output, step_pro = self.execute_step(prog_step, inspect)
                pro_str += step_pro + '\n'
            else:
                step_output = self.execute_step(prog_step, inspect)

        if isinstance(inspect, str) and inspect.casefold() in ['html', 'text']:
            return step_output, prog.state, pro_str.strip('\n')

        return step_output, prog.state


class ProgramGenerator:
    def __init__(self, prompter, temperature=0.7, top_p=0.5, prob_agg='mean'):
        openai.api_key = os.getenv("OPENAI_API_KEY")
        self.prompter = prompter
        self.temperature = temperature
        self.top_p = top_p
        self.prob_agg = prob_agg

    def compute_prob(self, response):
        eos = '<|endoftext|>'
        for i, token in enumerate(response.choices[0]['logprobs']['tokens']):
            if token == eos:
                break

        if self.prob_agg == 'mean':
            agg_fn = np.mean
        elif self.prob_agg == 'sum':
            agg_fn = np.sum
        else:
            raise NotImplementedError

        return np.exp(agg_fn(
            response.choices[0]['logprobs']['token_logprobs'][:i]))

    def generate(self, inputs):
        response = openai.Completion.create(
            model="gpt-3.5-turbo-instruct",
            prompt=self.prompter(inputs),
            temperature=self.temperature,
            max_tokens=512,
            top_p=self.top_p,
            frequency_penalty=0,
            presence_penalty=0,
            n=1,
            logprobs=1
        )

        prob = self.compute_prob(response)
        prog = response.choices[0]['text'].lstrip('\n').rstrip('\n')
        return prog, prob
