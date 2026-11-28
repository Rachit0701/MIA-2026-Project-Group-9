import os
import tempfile
import unittest

from pipeline import _default_data_dir


class DefaultDataDirectoryTests(unittest.TestCase):
    def test_prefers_conventional_data_layout(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            script_dir = os.path.join(temp_dir, 'project')
            conventional_dir = os.path.join(temp_dir, 'data')
            alternative_dir = os.path.join(temp_dir, 'MIAlab_data', 'MIALab_2020_Dataset')
            os.makedirs(script_dir)

            for root in (conventional_dir, alternative_dir):
                for name in ('atlas', 'train', 'test'):
                    os.makedirs(os.path.join(root, name))

            self.assertEqual(_default_data_dir(script_dir), conventional_dir)

    def test_uses_alternative_layout_when_conventional_data_is_missing(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            script_dir = os.path.join(temp_dir, 'project')
            alternative_dir = os.path.join(temp_dir, 'MIAlab_data', 'MIALab_2020_Dataset')
            os.makedirs(script_dir)

            for name in ('atlas', 'train', 'test'):
                os.makedirs(os.path.join(alternative_dir, name))

            self.assertEqual(_default_data_dir(script_dir), alternative_dir)


if __name__ == '__main__':
    unittest.main()
