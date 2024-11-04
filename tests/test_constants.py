import unittest

from wsssss.constants import pre15140
from wsssss.constants import post15140

class TestConstants(unittest.TestCase):
    def test_scalar(self):
        for c in [pre15140, post15140]:
            keys = list(c.__dict__.keys())
            i_start = keys.index('version')
            for key in keys[i_start:]:
                if key == 'version':
                    self.assertTrue(isinstance(c.version, str))
                else:
                    self.assertTrue(isinstance(c.__dict__[key], (float, int)), f'{key} is not int or float.')
