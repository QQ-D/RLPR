import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from utils.training_state import save_checkpoint_atomic


class AtomicCheckpointTests(unittest.TestCase):
    def test_successful_save_roundtrips_and_interrupted_write_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'epoch.pth.tar'
            save_checkpoint_atomic({'epoch': 1, 'value': torch.arange(3)}, path)
            old = path.read_bytes()
            self.assertEqual(torch.load(path)['epoch'], 1)

            def interrupted(value, handle):
                handle.write(b'partial checkpoint')
                raise OSError('simulated interrupted disk write')

            with patch('utils.training_state.torch.save', interrupted):
                with self.assertRaises(OSError):
                    save_checkpoint_atomic({'epoch': 2}, path)
                self.assertEqual(path.read_bytes(), old)
                absent = Path(directory) / 'new.pth.tar'
                with self.assertRaises(OSError):
                    save_checkpoint_atomic({'epoch': 2}, absent)
                self.assertFalse(absent.exists())
            self.assertEqual(list(Path(directory).iterdir()), [path])


if __name__ == '__main__':
    unittest.main()
