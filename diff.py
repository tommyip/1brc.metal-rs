#!/usr/bin/env python3

import sys
import re

if len(sys.argv) != 3:
    print('Usage: diff.py EXPECTED ACTUAL')
    sys.exit(-1)


def to_map(content: str):
    m = {}
    RE = r'(, |^)(.*?=-?\d{1,2}\.\d/-?\d{1,2}\.\d/-?\d{1,2}\.\d)'
    for (_, line) in re.findall(RE, content.strip('{}\n')):
        name, stats = line.split('=')
        m[name] = stats
    return m


with open(sys.argv[1]) as f:
    expected = to_map(f.read())

with open(sys.argv[2]) as f:
    actual = to_map(f.read())

missing = []

n_min_diff = 0
n_mean_diff = 0
n_max_diff = 0
for name, stat in list(expected.items()):
    if name in actual:
        act_min, act_mean, act_max = actual[name].split('/')
        stat_min, stat_mean, stat_max = stat.split('/')
        min_diff, mean_diff, max_diff = '', '', ''
        if act_min != stat_min:
            min_diff = f' min [{stat_min}:{act_min}]'
            n_min_diff += 1
        if act_mean != stat_mean:
            mean_diff = f' mean [{stat_mean}:{act_mean}]'
            n_mean_diff += 1
        if act_max != stat_max:
            max_diff = f' max [{stat_max}:{act_max}]'
            n_max_diff += 1
        if actual[name] != stat:
            print(f'{name}{min_diff}{mean_diff}{max_diff}')
        del actual[name]
    else:
        missing.append((name, stat))

if len(missing) > 0:
    print('\nMissing from actual:')
    for name, stat in missing:
        print(f'{name} {stat}')

if len(actual) > 0:
    print('\nExcess from actual:')
    for name, stat in actual.items():
        print(f'{name} {stat}')

print(
    f'total expected: {len(expected)}, min diff: {n_min_diff}, '
    f'mean diff: {n_mean_diff}, max diff: {n_max_diff}, '
    f'missing: {len(missing)}, excess: {len(actual)}')
