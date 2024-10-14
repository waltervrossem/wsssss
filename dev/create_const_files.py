#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re

try:
    mesa_dir = os.environ['MESA_DIR']
except KeyError:
    print('Need a MESA_DIR environment variable to create constants file.')
    raise

const_file = f'{mesa_dir}/const/public/const_def.f90'

with open(f'{mesa_dir}/data/version_number', 'r') as handle:
    version = handle.read().strip()

if not os.path.isfile(const_file):
    raise FileNotFoundError(const_file)
else:
    with open(const_file, 'r') as handle:
        const_file_str = handle.read()

lines = const_file_str.replace('\n         !', ' !').split('\n')
lines = [_.strip() for _ in lines if '::' in _]
lines = [_ for _ in lines if '=' in _]
lines = [_ for _ in lines if 'character' not in _]
lines = [_ for _ in lines if 'logical' not in _]
lines = [_ for _ in lines if 'intent' not in _]
lines = [_ for _ in lines if 'selected_' not in _]
lines = [_.split('::')[1].strip() for _ in lines]
lines = [_.replace('_dp ', ' ') for _ in lines]
lines = [_.replace('! =', ' =').replace('!', '#').replace(';', '#') for _ in lines]

for i, line in enumerate(lines):
    new_line = line
    if line.count('=') > 1:  # Fix multiple = in line.
        parts = line.split('=')
        new_line = f'{parts[0]}={parts[1]}# ={"=".join(parts[2:])}'

    if line == 'sige  = electron cross section for Thomson scattering':
        new_line = 'sige = 6.6524587158d-025 # Thomson scattering electron cross section'

    parts = new_line.split('#')
    new_line = f'{parts[0].lower()}#{"#".join(parts[1:])}'
    if new_line.endswith('#'):
        new_line = new_line[:-1]

    lines[i] = new_line


re_dbl_fort = re.compile(r'(\d)[dD]([-+]?\d)')
for i, line in enumerate(lines):
    num = line.split('=')[1].split('#')[0].strip()
    new_num = re_dbl_fort.sub(r'\1E\2', num)
    lines[i] = line.replace(num, new_num)

s = '#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n\n'
s += f"version = '{version}'  # MESA version used to generate constants.py.\n\n"
s += '\n'.join(lines)
s += '\n'
if version < '15140':
    fname = 'pre15140.py'
else:
    fname = 'post15140.py'
with open(f'../src/wssss/constants/{fname}', 'w') as handle:
    handle.write(s)
