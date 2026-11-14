"""Resume the original torch RandomSampler without changing uninterrupted order."""
from copy import deepcopy
import json
from pathlib import Path
import torch
from torch.utils.data import RandomSampler

VERSION = 'gad.random_sampler.v1'


class ExactRandomSampler(RandomSampler):
    def __iter__(self):
        return _ExactIterator(self)


class _ExactIterator:
    def __init__(self, sampler):
        self.sampler = sampler
        self.start = sampler.generator.get_state().clone()
        self.iterator = RandomSampler.__iter__(sampler)
        self.yielded = 0

    def __iter__(self):
        return self

    def __next__(self):
        value = next(self.iterator)
        self.yielded += 1
        return value

    def state_dict(self):
        return {'version': VERSION, 'epoch_start_generator_state': self.start.clone(),
                'samples_yielded': self.yielded, 'dataset_size': len(self.sampler.data_source),
                'replacement': self.sampler.replacement, 'num_samples': self.sampler.num_samples}

    def load_state_dict(self, state):
        if state.get('version') != VERSION:
            raise ValueError('exact sampler state missing; preserve this checkpoint and start a new run')
        expected = (len(self.sampler.data_source), self.sampler.replacement, self.sampler.num_samples)
        if tuple(state[k] for k in ('dataset_size', 'replacement', 'num_samples')) != expected:
            raise ValueError('exact sampler dataset/options changed')
        self.start = state['epoch_start_generator_state'].clone()
        self.sampler.generator.set_state(self.start)
        self.iterator = RandomSampler.__iter__(self.sampler)
        self.yielded = 0
        for _ in range(state['samples_yielded']):
            next(self)


def main_sampler_state(state):
    if '_snapshot' in state:
        return state['_snapshot']['_main_snapshot']['_sampler_iter_state']
    return state['_sampler_iter_state']


def prepare_state(loader, state, checkpoint):
    """Only explicit stage handoff may import a legacy cursor; resume fails closed."""
    if not isinstance(getattr(loader, 'sampler', None), ExactRandomSampler):
        return state
    state = deepcopy(state)
    leaf = main_sampler_state(state)
    saved = leaf.get('sampler_iter_state') or {}
    if saved.get('version') == VERSION:
        return state
    marker = Path(checkpoint) / 'handoff_initialization.json'
    handoff = json.loads(marker.read_text()) if marker.is_file() else {}
    if handoff.get('same_stage_resume') is not False:
        raise ValueError('This legacy checkpoint has no epoch-start sampler RNG state. '
                         'Exact same-stage resume is impossible; preserve it and use a new run '
                         'from an explicit Warmup handoff. No silent cursor-only fallback is allowed.')
    # Match the OLD Warmup -> GAD import exactly (seeded sampler + cursor skip),
    # rather than guessing the absent Warmup RNG state. Source data.pt stays intact.
    sampler = loader.sampler
    generator = torch.Generator().manual_seed(sampler.generator.initial_seed())
    leaf['sampler_iter_state'] = {'version': VERSION,
        'epoch_start_generator_state': generator.get_state(), 'samples_yielded': leaf['samples_yielded'],
        'dataset_size': len(sampler.data_source), 'replacement': sampler.replacement,
        'num_samples': sampler.num_samples}
    return state
