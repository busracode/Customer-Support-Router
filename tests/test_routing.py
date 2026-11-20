import json
from pathlib import Path
import unittest
from src.routing import TEAMS, route, small_talk, prepare_message

class RoutingTests(unittest.TestCase):
    def test_label_coverage(self):
        labels = json.loads((Path(__file__).resolve().parents[1] / "data/processed/labels.json").read_text())
        self.assertEqual(set(labels) - {"oos"}, set(TEAMS))

    def test_rejection(self):
        self.assertEqual(route("order_status", .2, .32)[0], "Human Agent")
        self.assertEqual(route("oos", .99, .32)[0], "Human Agent")
        self.assertEqual(route("order_status", .9, .32)[0], "Orders & Shopping")
        self.assertEqual(route("unexpected", .9, .32)[0], "Human Agent")

    def test_greeting_does_not_swallow_request(self):
        self.assertIsNotNone(small_talk("Hello!"))
        self.assertIsNone(small_talk("Hello, where is my order?"))
        self.assertIsNotNone(small_talk("I feel frustrated"))

    def test_optional_correction_and_codes(self):
        message = "My dishwahser DW-6534 is broken"
        self.assertIn("dishwahser", prepare_message(message)[0])
        text, codes, changes = prepare_message(message, True)
        self.assertIn("dishwasher", text)
        self.assertEqual(codes, ["DW-6534"])
        self.assertEqual(changes, [("dishwahser", "dishwasher")])

if __name__ == "__main__":
    unittest.main()
