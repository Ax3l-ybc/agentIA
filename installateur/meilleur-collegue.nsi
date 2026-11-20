; =============================================================================
;  Installateur Windows de « Meilleur collègue » (Mon Assistant)
;
;  Fabriqué avec NSIS (logiciel libre), sous Windows ou Linux :
;      makensis installateur/meilleur-collegue.nsi
;  ou simplement : installateur/construire.sh   (sous Linux ou macOS)
;
;  Ce que fait l'installateur :
;  - copie le programme dans %LOCALAPPDATA%\Programs\Meilleur collegue (pas besoin d'être
;    administrateur, donc il marche aussi sur un PC de travail) ;
;  - lance installer.bat, qui télécharge le reste (Python, llama.cpp, Gemma 4, la voix :
;    environ 4 Go, 10 à 30 minutes la première fois) ;
;  - crée le raccourci « Meilleur collègue » sur le Bureau et dans le menu Démarrer ;
;  - s'ajoute à « Applications installées » de Windows, pour une désinstallation propre.
;  Une mise à jour (nouvel installateur par-dessus) garde config.yaml et vos assistants.
; =============================================================================

Unicode true
!include "MUI2.nsh"
!include "LogicLib.nsh"

!define NOM "Meilleur collègue"
!define NOM_DOSSIER "Meilleur collegue"
!define CLE_DESINSTALLATION "Software\Microsoft\Windows\CurrentVersion\Uninstall\MeilleurCollegue"
!ifndef VERSION
  !define VERSION "1.0.0"
!endif
!define RACINE ".."

Name "${NOM}"
OutFile "Meilleur-collegue-installation-${VERSION}.exe"
InstallDir "$LOCALAPPDATA\Programs\${NOM_DOSSIER}"
InstallDirRegKey HKCU "${CLE_DESINSTALLATION}" "InstallLocation"
RequestExecutionLevel user
SetCompressor /SOLID lzma
BrandingText "${NOM} ${VERSION}"

VIProductVersion "${VERSION}.0"
VIAddVersionKey "ProductName" "${NOM}"
VIAddVersionKey "FileDescription" "Installation de ${NOM}"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "ProductVersion" "${VERSION}"
VIAddVersionKey "LegalCopyright" "Logiciel libre"

!define MUI_ICON "${RACINE}\assistant\static\meilleur-collegue.ico"
!define MUI_UNICON "${RACINE}\assistant\static\meilleur-collegue.ico"
!define MUI_ABORTWARNING

!define MUI_WELCOMEPAGE_TITLE "Installation de ${NOM}"
!define MUI_WELCOMEPAGE_TEXT "Votre assistant personnel, à l'écrit ou à la voix, qui tourne entièrement sur cet ordinateur.$\r$\n$\r$\nL'installation télécharge ensuite le « cerveau » (Gemma 4) et la voix : environ 4 Go, et 10 à 30 minutes la première fois. Gardez l'ordinateur branché à internet jusqu'à la fin.$\r$\n$\r$\nUne fenêtre noire montrera l'avancement : ne la fermez pas.$\r$\n$\r$\nCliquez sur Suivant pour continuer."
!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!define MUI_FINISHPAGE_TEXT "${NOM} est installé.$\r$\n$\r$\nPour le lancer plus tard : double-cliquez sur « ${NOM} » sur votre Bureau, ou cherchez-le dans le menu Démarrer."
!define MUI_FINISHPAGE_RUN
!define MUI_FINISHPAGE_RUN_TEXT "Lancer ${NOM} maintenant"
!define MUI_FINISHPAGE_RUN_FUNCTION Lancer
!insertmacro MUI_PAGE_FINISH

!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES

!insertmacro MUI_LANGUAGE "French"

Function Lancer
  ExecShell "open" "$INSTDIR\lancer.bat"
FunctionEnd

Section "${NOM}" SecPrincipale
  SectionIn RO
  ; Place nécessaire en tout : le programme, Python et ses modules, Gemma 4 et la voix.
  AddSize 6500000

  SetOutPath "$INSTDIR"
  File "${RACINE}\lancer.bat"
  File "${RACINE}\installer.bat"
  File "${RACINE}\creer_raccourci.bat"
  File "${RACINE}\requirements.txt"
  File "${RACINE}\README.md"
  File /r /x "__pycache__" /x "*.pyc" "${RACINE}\assistant"

  ; Vos réglages et vos assistants ne sont jamais écrasés par une mise à jour.
  SetOverwrite off
  File "${RACINE}\config.yaml"
  File /r /x "corbeille" "${RACINE}\assistants"
  SetOverwrite on

  ; Le reste (Python, llama.cpp, Gemma 4, la voix) se télécharge : fenêtre noire d'avancement.
  DetailPrint "Téléchargement et installation des composants (fenêtre noire, 10 à 30 minutes)…"
  ExecWait '"$SYSDIR\cmd.exe" /c ""$INSTDIR\installer.bat" /auto"' $0
  ${If} $0 != 0
    MessageBox MB_ICONEXCLAMATION|MB_OK "Une partie de l'installation n'a pas abouti (souvent : connexion internet coupée).$\r$\n$\r$\nRelancez simplement cet installateur : il reprendra là où il s'est arrêté."
  ${EndIf}

  ; Raccourcis : Bureau et menu Démarrer.
  CreateShortCut "$DESKTOP\${NOM}.lnk" "$INSTDIR\lancer.bat" "" "$INSTDIR\assistant\static\meilleur-collegue.ico" 0 SW_SHOWNORMAL "" "Lancer mon assistant personnel"
  CreateShortCut "$SMPROGRAMS\${NOM}.lnk" "$INSTDIR\lancer.bat" "" "$INSTDIR\assistant\static\meilleur-collegue.ico" 0 SW_SHOWNORMAL "" "Lancer mon assistant personnel"

  ; Désinstallation propre (Paramètres > Applications > Applications installées).
  WriteUninstaller "$INSTDIR\Desinstaller.exe"
  WriteRegStr HKCU "${CLE_DESINSTALLATION}" "DisplayName" "${NOM}"
  WriteRegStr HKCU "${CLE_DESINSTALLATION}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "${CLE_DESINSTALLATION}" "Publisher" "${NOM}"
  WriteRegStr HKCU "${CLE_DESINSTALLATION}" "DisplayIcon" "$INSTDIR\assistant\static\meilleur-collegue.ico"
  WriteRegStr HKCU "${CLE_DESINSTALLATION}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${CLE_DESINSTALLATION}" "UninstallString" '"$INSTDIR\Desinstaller.exe"'
  WriteRegDWORD HKCU "${CLE_DESINSTALLATION}" "NoModify" 1
  WriteRegDWORD HKCU "${CLE_DESINSTALLATION}" "NoRepair" 1
  WriteRegDWORD HKCU "${CLE_DESINSTALLATION}" "EstimatedSize" 6500000
SectionEnd

Section "Uninstall"
  Delete "$DESKTOP\${NOM}.lnk"
  Delete "$SMPROGRAMS\${NOM}.lnk"

  ; Les données (conversations, profils, modèles téléchargés) sont dans %USERPROFILE%\MonAssistant.
  MessageBox MB_ICONQUESTION|MB_YESNO|MB_DEFBUTTON2 "Supprimer aussi vos conversations, vos profils et les modèles téléchargés (dossier MonAssistant, environ 4 Go) ?$\r$\n$\r$\nRépondez Non pour les garder (par exemple pour réinstaller plus tard)." IDYES supprimer_donnees
    ; On garde aussi une copie de vos réglages et de vos assistants.
    CreateDirectory "$PROFILE\MonAssistant\sauvegarde-reglages"
    CopyFiles /SILENT "$INSTDIR\config.yaml" "$PROFILE\MonAssistant\sauvegarde-reglages"
    CopyFiles /SILENT "$INSTDIR\assistants" "$PROFILE\MonAssistant\sauvegarde-reglages"
    Goto supprimer_programme
  supprimer_donnees:
    RMDir /r "$PROFILE\MonAssistant"
  supprimer_programme:
  RMDir /r "$INSTDIR"
  DeleteRegKey HKCU "${CLE_DESINSTALLATION}"
SectionEnd
