import itertools
import json


def avg(l):
    return sum(l) / len(l)

def process_log_lists(l):
    return list(itertools.chain.from_iterable(l))


def save_json(data, path):
    with open(path, 'w') as file:
        json.dump(data, file, indent=4)


def risky_target(target, beta):
    ...