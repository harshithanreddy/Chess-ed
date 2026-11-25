import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("prefect"), "prefect not installed")
class FlowImport(unittest.TestCase):
    def test_flow_module_imports(self):
        from metalens import flow

        self.assertTrue(callable(flow.monthly_pipeline))


if __name__ == "__main__":
    unittest.main()
