import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const workbookPath = "/tmp/v4_promoted_corrected_alfred_baseline_audit_pack_2026-04-30.xlsx";
const input = await FileBlob.load(workbookPath);
const workbook = await SpreadsheetFile.importXlsx(input);

const ranges = [
  "README_manifest!A1:B12",
  "promoted_baseline_metrics!A1:I4",
  "trailing_metrics!A1:I6",
  "audit_checks!A1:C10",
];

for (const range of ranges) {
  const table = await workbook.inspect({
    kind: "table",
    range,
    include: "values,formulas",
    tableMaxRows: 20,
    tableMaxCols: 12,
  });
  console.log(`INSPECT ${range}`);
  console.log(table.ndjson);
}

const errors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 300 },
  summary: "formula error scan",
});
console.log("ERROR_SCAN");
console.log(errors.ndjson);

for (const sheetName of ["README_manifest", "promoted_baseline_metrics", "audit_checks"]) {
  const blob = await workbook.render({ sheetName, scale: 2 });
  console.log(`RENDER ${sheetName} ${blob.byteLength ?? blob.bytes?.length ?? 0}`);
}
