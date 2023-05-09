from dataclasses import dataclass
from typing import List
import yaml

@dataclass
class Scenario:
    room: str
    receptacles: List[str]
    seen_objects: List[str]
    seen_placements: List[List[str]]
    unseen_objects: List[str]
    unseen_placements: List[List[str]]
    annotator_notes: str
    tags: List[str]

def load_scenarios(path='scenarios.yml'):
    with open(path, 'r', encoding='utf8') as f:
        scenarios = list(map(lambda x: Scenario(**x), yaml.safe_load(f)))
    return scenarios

def parse_summary(summarization_completion):
    lines = [l for l in map(str.strip, summarization_completion.split('\n')) if len(l) > 0]
    if len(lines) > 1:
        print('Warning: Using first line of multi-line summary')
    return lines[0]

def parse_placements(placement_completion, objects):
    placements = []
    first_line = True
    for line in placement_completion.strip().split('\n'):
        if first_line:
            obj = objects[0]
            recep = line
            first_line = False
        else:
            if len(line) == 0:
                print('Warning: Stopping since newline was encountered')
                break
            placement_args = line.split(',')
            if len(placement_args) != 2:
                print('Warning: Skipping invalid placement')
                continue
            obj, recep = placement_args
            if '(' in obj:
                obj = obj.split('(')[1].strip().replace('"', '')
            else:
                print('Warning: Found possibly invalid placement')
                obj = obj.strip().replace('"', '')
        recep = recep.strip().replace(')', '').replace('"', '')
        placements.append([obj, recep])
    return placements

def check_placements(predicted_placements, correct_placements):
    correct_placements_dict = {}
    for obj, recep in correct_placements:
        correct_placements_dict[obj] = recep

    corrects = []
    for obj, recep in predicted_placements:  # Note that for repeats, this will only score the first instance
        corrects.append(obj in correct_placements_dict and recep == correct_placements_dict.pop(obj))

    accuracy = sum(corrects) / len(correct_placements)

    return corrects, accuracy

if __name__ == '__main__':
    assert check_placements([['o1', 'r1'], ['o2', 'r2']], [['o1', 'r1'], ['o2', 'r2']]) == ([True, True], 1.0)
    assert check_placements([['o1', 'r2'], ['o2', 'r2']], [['o1', 'r1'], ['o2', 'r2']]) == ([False, True], 0.5)
    assert check_placements([['o3', 'r1'], ['o2', 'r2']], [['o1', 'r1'], ['o2', 'r2']]) == ([False, True], 0.5)
    assert check_placements([['o1', 'r1'], ['o1', 'r1']], [['o1', 'r1'], ['o2', 'r2']]) == ([True, False], 0.5)
