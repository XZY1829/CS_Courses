# Export experiment report Markdown to PDF (Chinese requires xelatex + ctex)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$Report = Get-ChildItem -Path $Root -Filter "*Transformer*.md" | Select-Object -First 1
if (-not $Report) {
    throw "Report markdown not found: *Transformer*.md"
}
$InputMd = $Report.FullName
$OutputPdf = Join-Path $Root ($Report.BaseName + ".pdf")

Write-Host "Building PDF ..."
pandoc $InputMd `
    -o $OutputPdf `
    --pdf-engine=xelatex `
    --from markdown+yaml_metadata_block `
    --toc `
    --number-sections

if ($LASTEXITCODE -ne 0) {
    throw "Pandoc failed. Install MiKTeX ctex: mpm --install=ctex"
}

Write-Host "Done: $OutputPdf"
