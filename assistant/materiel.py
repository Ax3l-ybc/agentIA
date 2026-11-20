"""Ce que l'assistant sait du PC : mémoire vive et carte graphique.

    .venv\\Scripts\\python.exe -m assistant.materiel             affiche le matériel détecté
    .venv\\Scripts\\python.exe -m assistant.materiel --installer  (installer.bat) prépare llama.cpp
                                                                 pour la carte graphique s'il y en a une
"""

from __future__ import annotations

import ctypes
import dataclasses
import os
import platform
import re
import shutil
import subprocess
import sys


@dataclasses.dataclass
class Carte:
    nom: str
    memoire_go: float  # 0 si inconnue
    genre: str  # "nvidia" (CUDA) ou "vulkan" (AMD, Intel Arc…)


def ram_go() -> tuple[float, float]:
    """(mémoire vive totale, disponible), en Go."""
    try:
        if sys.platform == "win32":
            class Etat(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
            etat = Etat()
            etat.dwLength = ctypes.sizeof(Etat)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(etat))
            return etat.ullTotalPhys / 1e9, etat.ullAvailPhys / 1e9
        infos = {}
        with open("/proc/meminfo") as f:
            for ligne in f:
                cle, valeur = ligne.split(":", 1)
                infos[cle] = int(valeur.split()[0]) * 1024
        return infos["MemTotal"] / 1e9, infos.get("MemAvailable", 0) / 1e9
    except Exception:  # noqa: BLE001
        return 0.0, 0.0


def _commande(arguments: list[str], delai: float = 10) -> str:
    options = {"creationflags": subprocess.CREATE_NO_WINDOW} if sys.platform == "win32" else {}
    try:
        return subprocess.run(arguments, capture_output=True, text=True, timeout=delai, **options).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def cartes() -> list[Carte]:
    """Les cartes graphiques utilisables par llama.cpp (les puces intégrées sont ignorées : elles
    partagent la mémoire vive et n'accélèrent guère)."""
    trouvees = []
    if shutil.which("nvidia-smi"):
        for ligne in _commande(["nvidia-smi", "--query-gpu=name,memory.total",
                                "--format=csv,noheader,nounits"]).splitlines():
            if "," in ligne:
                nom, memoire = ligne.rsplit(",", 1)
                try:
                    trouvees.append(Carte(nom.strip(), round(float(memoire) / 1024, 1), "nvidia"))
                except ValueError:
                    continue
    if trouvees or sys.platform != "win32":
        return trouvees
    # Windows sans NVIDIA : une carte AMD Radeon RX ou Intel Arc marche avec Vulkan. Windows
    # indique mal leur mémoire (4 Go au plus) : on lit plutôt le registre du pilote.
    sortie = _commande(["powershell", "-NoProfile", "-Command",
                        "Get-ItemProperty 'HKLM:\\SYSTEM\\ControlSet001\\Control\\Class\\"
                        "{4d36e968-e325-11ce-bfc1-08002be10318}\\0*' -ErrorAction SilentlyContinue | "
                        "ForEach-Object { \"$($_.DriverDesc)|$($_.'HardwareInformation.qwMemorySize')\" }"])
    for ligne in sortie.splitlines():
        nom, _, memoire = ligne.partition("|")
        if re.search(r"Radeon RX|Radeon Pro|Arc\b|Arc\(TM\)", nom):
            go = round(int(memoire) / 1e9, 1) if memoire.strip().isdigit() else 0.0
            trouvees.append(Carte(nom.strip(), go, "vulkan"))
    return trouvees


def resume() -> dict:
    total, dispo = ram_go()
    liste = cartes()
    principale = max(liste, key=lambda c: c.memoire_go, default=None)
    return {
        "systeme": f"{platform.system()} {platform.release()}",
        "processeur": platform.processor() or platform.machine(),
        "coeurs": os.cpu_count() or 0,
        "ram_go": round(total, 1),
        "ram_disponible_go": round(dispo, 1),
        "cartes": [dataclasses.asdict(c) for c in liste],
        "vram_go": principale.memoire_go if principale else 0.0,
        "genre_carte": principale.genre if principale else None,
    }


def main() -> int:
    from . import llamacpp_gpu
    from .modeles import Choix, conseil, ecrire_choix, lire_choix, FICHIER_CHOIX

    infos = resume()
    print(f"Mémoire vive : {infos['ram_go']} Go ({infos['ram_disponible_go']} Go disponibles), "
          f"{infos['coeurs']} cœurs.")
    if infos["cartes"]:
        for c in infos["cartes"]:
            print(f"Carte graphique : {c['nom']} ({c['memoire_go'] or '?'} Go)")
    else:
        print("Pas de carte graphique utilisable : tout tournera sur le processeur.")
    if "--installer" not in sys.argv:
        return 0
    if infos["genre_carte"] and infos["vram_go"] >= 4:
        print(f"Téléchargement de llama.cpp pour la carte graphique ({infos['genre_carte']})…")
        try:
            chemin = llamacpp_gpu.installer(infos["genre_carte"], afficher=print)
            print(f"  -> {chemin}")
            if not FICHIER_CHOIX.exists():  # premier choix : on utilise la carte
                choix = lire_choix()
                ecrire_choix(Choix(choix.ecoute, choix.redacteur, carte_graphique=True))
        except Exception as exc:  # noqa: BLE001
            print(f"ATTENTION : llama.cpp pour la carte graphique n'a pas pu être installé ({exc}).")
            print("L'assistant marchera sur le processeur.")
    conseille = conseil(infos["ram_go"], infos["vram_go"])
    print(f"Cerveau conseillé pour ce PC : écoute {conseille.ecoute}"
          + (f", rédaction {conseille.redacteur}" if conseille.redacteur else "")
          + " (à choisir dans la page, bouton « Cerveau »).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
