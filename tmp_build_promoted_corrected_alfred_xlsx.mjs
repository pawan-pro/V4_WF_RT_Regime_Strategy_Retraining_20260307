import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const inputDir = "/tmp/v4_promoted_corrected_alfred_pack_data";
const outputPath = "/tmp/v4_promoted_corrected_alfred_baseline_audit_pack_2026-04-30.xlsx";
const sheets = [
  "README_manifest",
  "source_row_counts",
  "promoted_baseline_metrics",
  "spy_comparison",
  "annual_returns",
  "trailing_metrics",
  "equity_curve_daily_eod",
  "drawdown_daily_eod",
  "promotion_backup_manifest",
  "audit_checks",
];

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    const next = text[i + 1];
    if (quoted) {
      if (ch === '"' && next === '"') {
        cell += '"';
        i += 1;
      } else if (ch === '"') {
        quoted = false;
      } else {
        cell += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      row.push(cell);
      cell = "";
    } else if (ch === "\n") {
      row.push(cell);
      rows.push(row);
      row = [];
      cell = "";
    } else if (ch !== "\r") {
      cell += ch;
    }
  }
  if (cell.length || row.length) {
    row.push(cell);
    rows.push(row);
  }
  return rows;
}

function coerce(value) {
  if (value === "") return null;
  if (/^-?\d+(\.\d+)?([eE][+-]?\d+)?$/.test(value)) return Number(value);
  return value;
}

function colLetters(index) {
  let n = index + 1;
  let out = "";
  while (n > 0) {
    const rem = (n - 1) % 26;
    out = String.fromCharCode(65 + rem) + out;
    n = Math.floor((n - 1) / 26);
  }
  return out;
}

const workbook = Workbook.create();

for (const sheetName of sheets) {
  const csvPath = path.join(inputDir, `${sheetName}.csv`);
  const rows = parseCsv(await fs.readFile(csvPath, "utf8")).map((r) => r.map(coerce));
  const sheet = workbook.worksheets.add(sheetName);
  if (!rows.length) continue;
  const maxCols = Math.max(...rows.map((r) => r.length));
  const rectangular = rows.map((r) => {
    const copy = [...r];
    while (copy.length < maxCols) copy.push(null);
    return copy;
  });
  sheet.getRange("A1").write(rectangular);
  const header = sheet.getRange(`A1:${colLetters(maxCols - 1)}1`);
  header.format.fill = { type: "solid", color: "#163A5F" };
  header.format.font = { color: "#FFFFFF", bold: true };
  sheet.freezePanes.freezeRows(1);
  for (let c = 0; c < maxCols; c += 1) {
    const width = Math.min(
      Math.max(...rectangular.slice(0, 250).map((r) => String(r[c] ?? "").length), 10) * 7 + 16,
      320,
    );
    sheet.getRange(`${colLetters(c)}:${colLetters(c)}`).format.columnWidthPx = width;
  }
}

const output = await SpreadsheetFile.exportXlsx(workbook);
await output.save(outputPath);
console.log(outputPath);
