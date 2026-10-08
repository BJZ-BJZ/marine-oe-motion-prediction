"""OE motion prediction: verification entry points must refuse to run with assertions disabled."""
from pathlib import Path
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]

SCRIPTS = ['src/verify.py', 'src/verify_gru_gradients.py']

class VerificationGuards(unittest.TestCase):
    def test_optimized_entry_points_refuse_to_report_success(self):
        for rel in SCRIPTS:
            path = ROOT / rel
            with self.subTest(path=rel):
                result = subprocess.run([sys.executable, '-O', str(path)],
                                        capture_output=True, text=True)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('Verification requires assertions', result.stderr)


class Contracts(unittest.TestCase):
    def test_ridge_input_contract(self):
        import importlib.util
        import numpy as np
        spec = importlib.util.spec_from_file_location('inference', ROOT / 'src/inference.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        for x in [np.zeros((1, 20)), np.full((1, 21), np.nan)]:
            with self.assertRaises(ValueError):
                module.predict(x)

if __name__ == '__main__':
    unittest.main()
