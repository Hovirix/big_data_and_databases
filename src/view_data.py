"""Generate an offline viewer: nix develop --command uv run --locked python -m src.view_data"""

import hashlib
from html import escape
import json
from pathlib import Path
import re

import pandas as pd

from .overview import FIELDS, churn_summary


def create_view(source=Path("data/Telco-Customer-Churn.csv"), output=Path("outputs/data_view.html")):
    raw = pd.read_csv(source, keep_default_na=False)
    if set(raw.columns) != {field[0] for field in FIELDS} or raw.empty:
        raise ValueError("Expected the nonempty Telco dataset")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    settings_path = Path("outputs/modelling/settings.json")
    match = "No saved modelling fingerprint available."
    if settings_path.exists():
        expected = json.loads(settings_path.read_text())["source_sha256"]
        match = "✓ Matches the CSV used for modelling." if digest == expected else "⚠ DOES NOT match the CSV used for modelling."
    blanks = raw["TotalCharges"].astype(str).str.strip().eq("")
    churn = raw["Churn"].eq("Yes")
    grouped = raw.copy()
    grouped["Tenure (months)"] = pd.cut(
        pd.to_numeric(raw["tenure"]), [-1, 12, 24, 48, 72],
        labels=["0–12", "13–24", "25–48", "49–72"],
    )
    grouped["Monthly charges (quartiles)"] = pd.qcut(
        pd.to_numeric(raw["MonthlyCharges"]), 4,
    )
    names = ["Contract", "Tenure (months)", "Monthly charges (quartiles)"] + [
        name for name, _, kind, _ in FIELDS if kind == "Categorical" and name not in ["Churn", "Contract"]
    ]
    charts = []
    for name in names:
        summary = churn_summary(grouped, name)
        bars = "".join(
            f'<div class="bar"><span>{escape(str(row.category))}</span>'
            f'<meter min="0" max="100" value="{row.churn_percentage}">'
            f'{row.churn_percentage:.2f}%</meter>'
            f'<span>{row.churn_percentage:.2f}% <small>({row.churned:,}/{row.customers:,})</small></span></div>'
            for row in summary.itertuples()
        )
        charts.append(f'<section class="card"><h3>{escape(name)}</h3>{bars}</section>')
    dictionary = pd.DataFrame(FIELDS, columns=["Column", "Group", "Type", "Meaning"]).to_html(index=False)
    # Escape '<' inside JSON so data cannot terminate its script element.
    payload = json.dumps({
        "columns": raw.columns.tolist(), "rows": raw.values.tolist(),
    }, ensure_ascii=True).replace("<", "\\u003c")
    page = """<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Telco — data sanity check</title>
<style>
body{font:16px system-ui,sans-serif;margin:0;background:#f2f5f8;color:#182632}
main{max-width:1250px;margin:auto;padding:24px}h1{margin-bottom:8px}
a{color:#075b91}small,.muted{color:#526476}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(330px,1fr));gap:14px}
.card{background:white;border:1px solid #d9e1e8;border-radius:10px;padding:16px;margin:10px 0}
.stats{display:flex;gap:12px;flex-wrap:wrap}.stats .card{flex:1;min-width:150px}
.stats strong{display:block;font-size:28px}
.bar{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:12px 0}
.bar span:last-child{grid-column:1/-1;text-align:right}meter{width:100%;height:22px}
code{overflow-wrap:anywhere}.controls{display:flex;gap:12px;flex-wrap:wrap;align-items:center}
input,select,button{font:inherit;padding:8px}input[type=search]{width:min(100%,350px)}
button{cursor:pointer}button:disabled{cursor:default}
.scroll{overflow:auto;max-height:560px;border:1px solid #d9e1e8;margin:16px 0}
table{border-collapse:collapse;width:100%;font-size:14px;background:white}
th,td{border-bottom:1px solid #e0e6eb;padding:9px;text-align:left;white-space:nowrap}
th{position:sticky;top:0;background:#e6edf3}tr:hover td{background:#eef5fa}
.blank{background:#fff0bf;color:#704800}.yes{color:#a52725;font-weight:bold}
summary{cursor:pointer;font-weight:bold}footer{margin:24px 0}
</style>
<main>
<h1>Telco customer churn</h1>
<p>A quick visual sanity check — one row = one customer. No models run here.</p>
<p>Source: <a href="https://github.com/IBM/telco-customer-churn-on-icp4d">IBM Telco Customer Churn</a>.
Local file: <code>__SOURCE__</code></p>
<div class="card"><strong>__MATCH__</strong><br><small>SHA-256: <code>__HASH__</code></small></div>
<div class="stats">
<div class="card"><strong>__ROWS__</strong>customers</div>
<div class="card"><strong>__COLS__</strong>columns (ID + target + 19 candidates)</div>
<div class="card"><strong>__YES__%</strong>Churn = Yes (__YESCOUNT__ customers)</div>
<div class="card"><strong>__NO__%</strong>Churn = No</div>
<div class="card"><strong>__BLANKS__</strong>blank TotalCharges</div>
</div>
<h2>Original customer records</h2>
<p>No imputation or encoding. All CSV columns are shown; blank cells are highlighted.
Search matches any column. Filters below affect only this table, not the charts.</p>
<div class="controls">
<label>Search <input type="search" id="search" placeholder="Customer ID, Fiber optic, …"></label>
<label>Churn <select id="churn"><option value="">All</option><option>Yes</option><option>No</option></select></label>
<label><input type="checkbox" id="blanks"> Only blank TotalCharges</label>
</div>
<p id="count" role="status" aria-live="polite"></p>
<div class="scroll" tabindex="0" aria-label="Customer data, scroll horizontally for all columns">
<table><thead id="head"></thead><tbody id="body"></tbody></table></div>
<div class="controls"><button id="prev">← Previous</button><button id="next">Next →</button>
<span>100 records per page</span></div>
<noscript>Enable JavaScript to browse the records. Charts and the data dictionary work without it.</noscript>
<h2>Churn across groups — full dataset</h2>
<p>All bars use a 0–100% scale. Percentage = churners / customers in that group.
Parentheses show churners / group size, not the share of all churners.
Monthly charges use quartiles; interval notation (a, b] means &gt;a and ≤b.</p>
<div class="cards">__CHARTS__</div>
<details class="card"><summary>Data dictionary — all 21 columns</summary>
<div class="scroll">__DICTIONARY__</div></details>
<footer class="muted">Subscriptions ≠ usage intensity. These are descriptive associations, not causal effects.
This page embeds the CSV as a local snapshot. Regenerate it if the CSV changes.
No external scripts, server, internet connection or new dependencies required.</footer>
</main>
<script id="dataset" type="application/json">__DATA__</script>
<script>
const data = JSON.parse(document.getElementById('dataset').textContent);
const get = id => document.getElementById(id);
const churnIndex = data.columns.indexOf('Churn'), chargesIndex = data.columns.indexOf('TotalCharges');
const header = document.createElement('tr');
for (const name of data.columns) {
  const th = document.createElement('th'); th.textContent = name; th.scope = 'col'; header.append(th);
}
get('head').append(header);
let page = 0;
function render() {
  const query = get('search').value.trim().toLowerCase(), target = get('churn').value;
  const rows = data.rows.filter(row =>
    (!target || row[churnIndex] === target) &&
    (!get('blanks').checked || String(row[chargesIndex]).trim() === '') &&
    (!query || row.some(value => String(value).toLowerCase().includes(query))));
  const pages = Math.max(1, Math.ceil(rows.length / 100));
  page = Math.min(page, pages - 1);
  get('body').replaceChildren();
  for (const row of rows.slice(page * 100, (page + 1) * 100)) {
    const tr = document.createElement('tr');
    row.forEach((value, i) => {
      const td = document.createElement('td'), blank = String(value).trim() === '';
      td.textContent = blank ? '(blank)' : String(value);
      if (blank) td.className = 'blank';
      if (i === churnIndex && value === 'Yes') td.className = 'yes';
      tr.append(td);
    });
    get('body').append(tr);
  }
  get('count').textContent = `${rows.length.toLocaleString()} / ${data.rows.length.toLocaleString()} customers · Page ${page + 1} of ${pages}`;
  get('prev').disabled = page === 0; get('next').disabled = page === pages - 1;
}
for (const id of ['search', 'churn', 'blanks']) get(id).addEventListener('input', () => {page = 0; render();});
get('prev').addEventListener('click', () => {page--; render();});
get('next').addEventListener('click', () => {page++; render();});
render();
</script>
</html>
"""
    replacements = {
        "SOURCE": escape(str(source)), "MATCH": escape(match), "HASH": digest,
        "ROWS": f"{len(raw):,}", "COLS": str(len(raw.columns)),
        "YES": f"{churn.mean() * 100:.2f}", "NO": f"{(~churn).mean() * 100:.2f}",
        "YESCOUNT": f"{churn.sum():,}", "BLANKS": str(blanks.sum()),
        "CHARTS": "".join(charts), "DICTIONARY": dictionary, "DATA": payload,
    }
    # One substitution pass: inserted CSV values are never treated as template tokens.
    page = re.sub(r"__(\w+)__", lambda match: replacements[match[1]], page)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(page, encoding="utf-8")
    return output


if __name__ == "__main__":
    print(f"Open in your browser → {create_view().resolve()}")
