# my-scribe — transcription audio/vidéo → texte (FR)

Transcription locale de fichiers **audio ou vidéo** en texte et sous-titres, optimisée pour le français.

- **Moteur** : [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2) — ~4× plus rapide et 2× moins de RAM que `openai-whisper`, quantization `int8` sur CPU.
- **Modèle FR par défaut** : `bofenghuang/whisper-large-v3-french` (fine-tune français complet : fidélité maximale).
- **CLI** : [typer](https://typer.tiangolo.com/) — options auto-documentées (`--help`).
- **Logs** : [loguru](https://github.com/Delgan/loguru).
- **Gestion d'environnement** : [uv](https://docs.astral.sh/uv/).

## Prérequis

- [uv](https://docs.astral.sh/uv/) installé.
- **Python ≥ 3.14** : récupéré automatiquement par `uv` via le fichier `.python-version`.
- Aucune installation de `ffmpeg` ni de `torch` requise : PyAV (embarqué par faster-whisper) décode l'audio/vidéo.

## Installation

```powershell
uv sync
```

`uv` crée un `.venv` local et installe les dépendances depuis `pyproject.toml`.

## Organisation des dossiers

| Dossier | Rôle |
|---------|------|
| `input/`  | Déposez ici vos fichiers audio/vidéo à transcrire. |
| `output/` | Les transcriptions (`.txt`, `.vtt`…) y sont écrites. |
| `models/` | Cache local des modèles téléchargés (volumineux). |

Ces trois dossiers sont ignorés par git (voir `.gitignore`).

## Utilisation

```powershell
# Traite tous les médias présents dans ./input → sorties dans ./output
uv run transcribe.py

# Un fichier précis
uv run transcribe.py input\Enregistrement.m4a

# Choisir les formats de sortie
uv run transcribe.py input\reunion.mp3 --formats txt,vtt,srt,json

# Détection automatique de langue + modèle multilingue
uv run transcribe.py input\interview.wav --lang auto --model large-v3

# Largeur de ligne du .txt (défaut 120 ; 0 = pas de coupure)
uv run transcribe.py input\note.m4a --line-length 80

# Amorce pour guider la ponctuation et le vocabulaire métier
uv run transcribe.py input\cours.m4a --initial-prompt "Cours de L1 : décrochage, human in the loop."

# Aide complète
uv run transcribe.py --help
```

## Options

| Option | Défaut | Rôle |
|--------|--------|------|
| `input` (positionnel) | `./input` | Fichier audio/vidéo, ou dossier à traiter en lot. |
| `--lang`, `-l` | `fr` | Langue (`fr`, `en`, `es`…) ou `auto` pour détection. |
| `--model`, `-m` | *selon langue* | Modèle faster-whisper. Défaut : fine-tune FR complet si `--lang fr`, sinon `large-v3`. |
| `--device` | `auto` | `cpu`, `cuda` ou `auto` (détecte le GPU, bascule CPU si libs CUDA absentes). |
| `--compute-type` | `auto` | `int8` (CPU), `float16` (GPU), `float32`, ou `auto`. |
| `--formats`, `-f` | `txt,vtt` | Formats séparés par des virgules : `txt`, `vtt`, `srt`, `json`. |
| `--output-dir`, `-o` | `./output` | Dossier de sortie. |
| `--line-length`, `-w` | `120` | Largeur max des lignes du `.txt` (`0` = aucune coupure). |
| `--beam-size` | `5` | Taille du beam search (qualité vs vitesse). |
| `--vad / --no-vad` | activé | Filtre VAD : supprime les silences, réduit les hallucinations. |
| `--initial-prompt` | — | Amorce ponctuation/vocabulaire (termes métier, style). |
| `--context / --no-context` | `--no-context` | Conditionne sur le texte précédent (cohérence ++, mais favorise les boucles). |
| `--paragraph-gap` | `1.5` | Pause (s) déclenchant un nouveau paragraphe dans le `.txt`. |
| `--models-dir` | `./models` | Cache local des modèles fine-tunés téléchargés. |
| `--force` | désactivé | Régénère même si les sorties existent déjà. |
| `--verbose`, `-v` | désactivé | Logs détaillés (progression, DEBUG). |

## Formats de sortie

- **`.txt`** : texte en **paragraphes** (découpés aux pauses > `--paragraph-gap`), lignes limitées à `--line-length` caractères.
- **`.vtt`** : sous-titres WebVTT horodatés.
- **`.srt`** : sous-titres SubRip horodatés.
- **`.json`** : segments bruts (`start`, `end`, `text`) + langue et durée détectées.

## Formats média pris en charge

**Audio** : `.m4a`, `.mp3`, `.wav`, `.flac`, `.aac`, `.ogg`, `.opus`, `.wma`
**Vidéo** : `.mp4`, `.mkv`, `.avi`, `.mov`, `.webm`, `.flv`, `.wmv`

## Modèles

Le fine-tune français est téléchargé automatiquement au premier usage dans `--models-dir`
(sous-dossier `ctranslate2/`). Variantes disponibles chez
[bofenghuang](https://huggingface.co/bofenghuang), à passer via `--model` :

| Modèle | Compromis |
|--------|-----------|
| `bofenghuang/whisper-large-v3-french` | **Défaut** — fidélité maximale, plus lent. |
| `bofenghuang/whisper-large-v3-french-distil-dec16` | Qualité proche, un peu plus rapide. |
| `bofenghuang/whisper-large-v3-french-distil-dec8` | Plus rapide, qualité moindre. |
| `bofenghuang/whisper-large-v3-french-distil-dec2` | Le plus rapide, qualité la plus basse. |

```powershell
uv run transcribe.py input\cours.m4a --model bofenghuang/whisper-large-v3-french-distil-dec2
```

## Développement / Qualité du code

Les outils de qualité sont déclarés dans le `dependency-group` **dev** de `pyproject.toml`
(non installés en production) et installés par `uv sync`.

| Outil | Rôle |
|-------|------|
| [Ruff](https://docs.astral.sh/ruff/) | Linter **+** formateur (remplace flake8, isort, black, pyupgrade…). |
| [mypy](https://mypy-lang.org/) | Vérification de types statique. |
| [pytest](https://docs.pytest.org/) | Tests unitaires des fonctions pures (`tests/`). |
| [pre-commit](https://pre-commit.com/) | Hooks git : lance ruff + mypy avant chaque commit, pytest avant chaque push. |

```powershell
# Installer les dépendances (dont le dev group) + les hooks git (une fois)
uv sync
uv run pre-commit install

# Contrôles manuels
uv run ruff check .          # lint
uv run ruff format .         # formatage
uv run mypy transcribe.py    # types
uv run pytest                # tests
```

Ces mêmes contrôles sont rejoués automatiquement en **intégration continue**
([GitHub Actions](.github/workflows/ci.yml)) à chaque push ou pull request vers `develop` et `main`.
