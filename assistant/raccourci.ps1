# Crée sur le Bureau le raccourci « Meilleur collègue », qui lance lancer.bat.
# Appelé par installer.bat et par creer_raccourci.bat.
$dossier = Split-Path -Parent $PSScriptRoot
$bureau = [Environment]::GetFolderPath('Desktop')
$chemin = Join-Path $bureau 'Meilleur collègue.lnk'
$raccourci = (New-Object -ComObject WScript.Shell).CreateShortcut($chemin)
$raccourci.TargetPath = Join-Path $dossier 'lancer.bat'
$raccourci.WorkingDirectory = $dossier
$raccourci.IconLocation = (Join-Path $dossier 'assistant\static\meilleur-collegue.ico') + ',0'
$raccourci.Description = 'Lancer mon assistant personnel'
$raccourci.Save()
Write-Host "Raccourci cree sur le Bureau : $chemin"
