"""Colab'a taşınan verinin bütünlüğü ve test verisinin dışarıda kalması."""

import ast
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.train_bert import validate_data


class ColabPackageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="swe015_colab_test_")
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name)
        with zipfile.ZipFile(ROOT / "colab/bert_training_bundle.zip") as archive:
            self.names = archive.namelist()
            archive.extractall(self.path)

    def test_original_training_data_and_no_test_upload(self):
        manifest, labels, data = validate_data(self.path)
        self.assertEqual(len(labels), 151)
        self.assertEqual(len(data["train"]), 15095)
        self.assertEqual(len(data["validation"]), 3100)
        self.assertNotIn("data/processed/test.jsonl", self.names)
        self.assertEqual(set(self.names), set(manifest["files"]) | {"colab_manifest.json"})

    def test_changed_data_rejected(self):
        with (self.path / "data/processed/train.jsonl").open("ab") as stream:
            stream.write(b"\n")
        with self.assertRaisesRegex(ValueError, "ozeti uyusmuyor"):
            validate_data(self.path)

    def test_notebook_and_bundled_python_syntax(self):
        notebook = json.loads((ROOT / "notebooks/03_bert_colab.ipynb").read_text(encoding="utf-8"))
        self.assertEqual(notebook["nbformat"], 4)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                ast.parse("".join(cell["source"]))
                self.assertEqual(cell["outputs"], [])
        for path in self.path.rglob("*.py"):
            ast.parse(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
