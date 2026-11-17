param([string]$Pptx, [string]$OutDir, [int]$Width = 1200)
# Export every slide of $Pptx to PNG in $OutDir via PowerPoint (read-only, no window)
New-Item -ItemType Directory -Force $OutDir | Out-Null
Get-ChildItem $OutDir -Filter *.png | Remove-Item -Force
$pp = New-Object -ComObject PowerPoint.Application
try {
  $p = $pp.Presentations.Open($Pptx, $true, $false, $false)
  $h = [int]($Width * $p.PageSetup.SlideHeight / $p.PageSetup.SlideWidth)
  foreach ($s in $p.Slides) { $s.Export((Join-Path $OutDir ("s{0:D2}.png" -f $s.SlideIndex)), "PNG", $Width, $h) }
  "exported $($p.Slides.Count) slides"
  $p.Close()
} finally { $pp.Quit() }
