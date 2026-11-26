import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

from src.view_data import create_view


class DatasetParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_dataset = False
        self.payload = ""

    def handle_starttag(self, tag, attrs):
        if tag == "script" and dict(attrs).get("id") == "dataset":
            self.in_dataset = True

    def handle_endtag(self, tag):
        if tag == "script":
            self.in_dataset = False

    def handle_data(self, text):
        if self.in_dataset:
            self.payload += text


class DataViewTest(unittest.TestCase):
    def test_embeds_original_data_and_preserves_source(self):
        source = Path("data/Telco-Customer-Churn.csv")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        with TemporaryDirectory() as directory:
            output = create_view(source, Path(directory) / "viewer.html")
            html = output.read_text()
            parser = DatasetParser()
            parser.feed(html)
            data = json.loads(parser.payload)
            raw = pd.read_csv(source, keep_default_na=False)
            self.assertEqual(data["columns"], raw.columns.tolist())
            self.assertEqual(data["rows"], raw.values.tolist())
            self.assertEqual(len(data["rows"]), 7043)
            charges = data["columns"].index("TotalCharges")
            self.assertEqual(sum(str(row[charges]).strip() == "" for row in data["rows"]), 11)
            self.assertIn(digest, html)
            self.assertIn("42.71%", html)
            self.assertIn('max="100"', html)
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), digest)

    def test_csv_text_cannot_terminate_json_script(self):
        with TemporaryDirectory() as directory:
            raw = pd.read_csv("data/Telco-Customer-Churn.csv", keep_default_na=False).head(20)
            text = "</script><script>alert(1)</script>__ROWS__"
            raw.loc[0, "customerID"] = text
            source = Path(directory) / "sample.csv"
            raw.to_csv(source, index=False)
            output = create_view(source, Path(directory) / "viewer.html")
            html = output.read_text()
            self.assertNotIn(text, html)
            parser = DatasetParser()
            parser.feed(html)
            self.assertEqual(json.loads(parser.payload)["rows"][0][0], text)
