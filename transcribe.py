"""Transcription audio/vidéo → texte (français par défaut) via faster-whisper.

Stack : faster-whisper (moteur), typer (CLI), loguru (logs). Géré par uv.

Exemples :
    uv run transcribe.py Enregistrement.m4a
    uv run transcribe.py ./mes-audios --formats txt,vtt,srt
    uv run transcribe.py reunion.mp3 --lang auto --model large-v3
    uv run transcribe.py cours.m4a --initial-prompt "Cours de L1, décrochage, human in the loop."

Voir toutes les options : uv run transcribe.py --help
"""

from __future__ import annotations

import json
import sys
import textwrap
import time
from pathlib import Path

import typer
from loguru import logger

# ----------------------------------------------------------------------------
# Constantes
# ----------------------------------------------------------------------------
VIDEO_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv"}
AUDIO_EXTENSIONS = {".m4a", ".mp3", ".wav", ".flac", ".aac", ".ogg", ".opus", ".wma"}
MEDIA_EXTENSIONS = VIDEO_EXTENSIONS | AUDIO_EXTENSIONS

# Modèle utilisé par défaut selon la langue.
# fr → fine-tune français complet (fidélité maximale) ; sinon large-v3 multilingue.
DEFAULT_MODEL_BY_LANG = {"fr": "bofenghuang/whisper-large-v3-french"}
DEFAULT_MODEL = "large-v3"

VALID_FORMATS = {"txt", "vtt", "srt", "json"}

app = typer.Typer(add_completion=False, help=__doc__)


# ----------------------------------------------------------------------------
# Logs
# ----------------------------------------------------------------------------
def configure_logging(verbose: bool) -> None:
    """Configure loguru : sortie console lisible, niveau DEBUG si --verbose."""
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if verbose else "INFO",
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <7}</level> | {message}",
    )


# ----------------------------------------------------------------------------
# Résolution device / compute / modèle
# ----------------------------------------------------------------------------
def resolve_device(device: str) -> str:
    """Détermine le device effectif (cuda si dispo, sinon cpu) sans dépendre de torch."""
    if device != "auto":
        return device
    try:
        import ctranslate2

        return "cuda" if ctranslate2.get_cuda_device_count() > 0 else "cpu"
    except Exception:
        return "cpu"


def resolve_compute_type(compute_type: str, device: str) -> str:
    """int8 sur CPU (rapide, léger), float16 sur GPU. Sinon respecte le choix explicite."""
    if compute_type != "auto":
        return compute_type
    return "float16" if device == "cuda" else "int8"


def build_model(model_path: str, device: str, compute_type: str, allow_cpu_fallback: bool):
    """Charge le modèle et le « chauffe » sur 1 s de silence.

    Sur `--device auto`, si le GPU est détecté mais que ses DLL CUDA/cuDNN manquent,
    on bascule proprement sur CPU/int8 au lieu de planter en pleine transcription.
    Retourne (model, device_effectif, compute_effectif).
    """
    import numpy as np
    from faster_whisper import WhisperModel

    def _load(dev: str, ct: str):
        model = WhisperModel(model_path, device=dev, compute_type=ct)
        # Warmup : force l'usage des libs GPU dès maintenant (échoue vite si absentes).
        list(model.transcribe(np.zeros(16_000, dtype=np.float32), language="fr")[0])
        return model

    try:
        return _load(device, compute_type), device, compute_type
    except RuntimeError as exc:
        if allow_cpu_fallback and device == "cuda":
            logger.warning(f"GPU indisponible ({exc}). Bascule automatique sur CPU/int8.")
            return _load("cpu", "int8"), "cpu", "int8"
        raise


def resolve_model(model_ref: str, models_dir: Path) -> str:
    """Retourne un chemin/identifiant chargeable par faster-whisper.

    Les modèles bofenghuang stockent le format CTranslate2 dans un sous-dossier
    `ctranslate2/`. On le télécharge en cache local puis on pointe dessus.
    Les tailles standard (large-v3, medium...) et autres repos CT2 passent tels quels.
    """
    if model_ref.startswith("bofenghuang/"):
        from huggingface_hub import snapshot_download

        local_dir = models_dir / model_ref.split("/")[-1]
        ct2_dir = local_dir / "ctranslate2"
        if not (ct2_dir / "model.bin").exists():
            logger.info(f"Téléchargement du modèle {model_ref} (sous-dossier ctranslate2)…")
            snapshot_download(
                repo_id=model_ref,
                local_dir=str(local_dir),
                allow_patterns="ctranslate2/*",
            )
        return str(ct2_dir)
    return model_ref


# ----------------------------------------------------------------------------
# Découverte des fichiers
# ----------------------------------------------------------------------------
def discover_media(input_path: Path) -> list[Path]:
    """Retourne la liste des fichiers média à traiter (fichier unique ou dossier)."""
    if input_path.is_file():
        return [input_path]
    files = sorted(f for f in input_path.iterdir() if f.suffix.lower() in MEDIA_EXTENSIONS)
    return files


# ----------------------------------------------------------------------------
# Écriture des sorties
# ----------------------------------------------------------------------------
def fmt_ts(seconds: float, sep: str = ".") -> str:
    """Formate un temps en HH:MM:SS<sep>mmm (sep='.' pour VTT, ',' pour SRT)."""
    ms = round(seconds * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1_000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


def group_paragraphs(segments: list[dict], gap: float, line_length: int) -> str:
    """Regroupe les segments en paragraphes ; nouveau paragraphe si la pause > gap (s).

    Si line_length > 0, chaque paragraphe est enveloppé à cette largeur (coupure aux
    espaces). line_length = 0 : une seule ligne par paragraphe.
    """
    paragraphs: list[str] = []
    buffer: list[str] = []
    for i, seg in enumerate(segments):
        buffer.append(seg["text"])
        next_start = segments[i + 1]["start"] if i + 1 < len(segments) else None
        if next_start is not None and (next_start - seg["end"]) > gap:
            paragraphs.append(" ".join(buffer))
            buffer = []
    if buffer:
        paragraphs.append(" ".join(buffer))
    if line_length > 0:
        paragraphs = [textwrap.fill(p, width=line_length) for p in paragraphs]
    return "\n\n".join(paragraphs)


def render_vtt(segments: list[dict]) -> str:
    lines = ["WEBVTT", ""]
    for seg in segments:
        lines.append(f"{fmt_ts(seg['start'])} --> {fmt_ts(seg['end'])}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def render_srt(segments: list[dict]) -> str:
    lines: list[str] = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{fmt_ts(seg['start'], ',')} --> {fmt_ts(seg['end'], ',')}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def render(
    fmt: str, segments: list[dict], info: dict, paragraph_gap: float, line_length: int
) -> str:
    if fmt == "txt":
        return group_paragraphs(segments, paragraph_gap, line_length) + "\n"
    if fmt == "vtt":
        return render_vtt(segments)
    if fmt == "srt":
        return render_srt(segments)
    if fmt == "json":
        payload = {
            "language": info["language"],
            "language_probability": info["language_probability"],
            "duration": info["duration"],
            "segments": segments,
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)
    raise ValueError(f"Format inconnu : {fmt}")


def build_output_path(media: Path, out_dir: Path | None, fmt: str, use_full_name: bool) -> Path:
    """Chemin du fichier de sortie. use_full_name conserve l'extension source (anti-collision)."""
    stem = media.name if use_full_name else media.stem
    directory = out_dir if out_dir else media.parent
    return directory / f"{stem}.{fmt}"


# ----------------------------------------------------------------------------
# Traitement d'un fichier
# ----------------------------------------------------------------------------
def process_one(
    model,
    media: Path,
    *,
    lang: str | None,
    formats: list[str],
    out_dir: Path | None,
    beam_size: int,
    vad: bool,
    initial_prompt: str | None,
    condition_previous: bool,
    paragraph_gap: float,
    line_length: int,
    use_full_name: bool,
    force: bool,
) -> str:
    """Transcrit un fichier et écrit les formats demandés. Retourne 'ok' ou 'skip'."""
    targets = {fmt: build_output_path(media, out_dir, fmt, use_full_name) for fmt in formats}

    if not force and all(t.exists() for t in targets.values()):
        logger.info(f"⏭  {media.name} : sorties déjà présentes, ignoré (--force pour régénérer).")
        return "skip"

    logger.info(f"🎧 Transcription de {media.name}…")
    t0 = time.perf_counter()

    segments_iter, info = model.transcribe(
        str(media),
        language=lang,
        beam_size=beam_size,
        vad_filter=vad,
        initial_prompt=initial_prompt,
        condition_on_previous_text=condition_previous,
    )

    logger.info(
        f"   langue={info.language} (p={info.language_probability:.2f}) | "
        f"durée audio={info.duration:.0f}s"
    )

    segments: list[dict] = []
    last_pct = 0
    for seg in segments_iter:  # génère la transcription au fil de l'itération
        segments.append({"start": seg.start, "end": seg.end, "text": seg.text.strip()})
        if info.duration:
            pct = int(seg.end / info.duration * 100)
            if pct >= last_pct + 20:
                last_pct = pct
                logger.debug(f"   … {pct}%")

    info_dict = {
        "language": info.language,
        "language_probability": info.language_probability,
        "duration": info.duration,
    }

    directory = out_dir if out_dir else media.parent
    directory.mkdir(parents=True, exist_ok=True)
    for fmt, target in targets.items():
        target.write_text(
            render(fmt, segments, info_dict, paragraph_gap, line_length), encoding="utf-8"
        )

    elapsed = time.perf_counter() - t0
    speed = (info.duration / elapsed) if elapsed else 0
    logger.success(
        f"✅ {media.name} → {', '.join(t.name for t in targets.values())} "
        f"({elapsed:.0f}s, ×{speed:.1f} temps réel)"
    )
    return "ok"


# ----------------------------------------------------------------------------
# Commande CLI
# ----------------------------------------------------------------------------
@app.command()
def main(
    input: Path = typer.Argument(
        Path("input"),
        help="Fichier audio/vidéo, ou dossier à traiter (défaut : ./input).",
    ),
    lang: str = typer.Option(
        "fr", "--lang", "-l", help="Langue (fr, en, es...) ou 'auto' pour détection."
    ),
    model: str | None = typer.Option(
        None,
        "--model",
        "-m",
        help="Modèle faster-whisper. Défaut : fine-tune FR si --lang fr, sinon large-v3.",
    ),
    device: str = typer.Option("auto", "--device", help="cpu, cuda ou auto."),
    compute_type: str = typer.Option(
        "auto", "--compute-type", help="int8, float16, float32 ou auto (int8 sur CPU)."
    ),
    formats: str = typer.Option(
        "txt,vtt", "--formats", "-f", help="Formats de sortie (virgules) : txt, vtt, srt, json."
    ),
    output_dir: Path = typer.Option(
        Path("output"), "--output-dir", "-o", help="Dossier de sortie (défaut : ./output)."
    ),
    beam_size: int = typer.Option(5, "--beam-size", help="Taille du beam search."),
    vad: bool = typer.Option(
        True, "--vad/--no-vad", help="Filtre VAD : supprime les silences (moins d'hallucinations)."
    ),
    initial_prompt: str | None = typer.Option(
        None,
        "--initial-prompt",
        help="Amorce pour guider ponctuation et vocabulaire métier.",
    ),
    condition_previous: bool = typer.Option(
        False,
        "--context/--no-context",
        help="Conditionner sur le texte précédent (cohérence ++, mais favorise les boucles).",
    ),
    paragraph_gap: float = typer.Option(
        1.5, "--paragraph-gap", help="Pause (s) marquant un nouveau paragraphe dans le .txt."
    ),
    line_length: int = typer.Option(
        120, "--line-length", "-w", help="Largeur max des lignes du .txt (0 = aucune coupure)."
    ),
    models_dir: Path = typer.Option(
        Path("./models"), "--models-dir", help="Cache local des modèles fine-tunés téléchargés."
    ),
    force: bool = typer.Option(False, "--force", help="Régénère même si les sorties existent."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Logs détaillés (DEBUG)."),
) -> None:
    """Transcrit un fichier ou un dossier audio/vidéo en texte (et sous-titres)."""
    configure_logging(verbose)

    # Validation des formats
    formats_list = [f.strip().lower() for f in formats.split(",") if f.strip()]
    invalid = set(formats_list) - VALID_FORMATS
    if invalid:
        logger.error(
            f"Format(s) invalide(s) : {', '.join(sorted(invalid))}. "
            f"Valides : {', '.join(sorted(VALID_FORMATS))}."
        )
        raise typer.Exit(1)

    # Découverte des fichiers
    if not input.exists():
        logger.error(
            f"Chemin introuvable : {input}. "
            f"Déposez des fichiers dans ./input ou précisez un chemin."
        )
        raise typer.Exit(1)
    media_files = discover_media(input)
    if not media_files:
        logger.error("Aucun fichier audio/vidéo trouvé.")
        raise typer.Exit(1)
    logger.info(f"{len(media_files)} fichier(s) à transcrire.")

    # Anti-collision : si deux sources ont le même nom (base), on garde l'extension source
    stems = [m.stem for m in media_files]
    use_full_name = len(set(stems)) != len(stems)
    if use_full_name:
        logger.warning("Noms en collision : les sorties conserveront l'extension source.")

    # Langue : 'auto' → détection (language=None)
    lang_arg: str | None = None if lang.lower() == "auto" else lang

    # Modèle par défaut selon la langue
    model_ref = model or DEFAULT_MODEL_BY_LANG.get(lang.lower(), DEFAULT_MODEL)
    dev = resolve_device(device)
    ct = resolve_compute_type(compute_type, dev)
    logger.info(f"Modèle : {model_ref} | device={dev} | compute={ct}")

    model_path = resolve_model(model_ref, models_dir)

    t0 = time.perf_counter()
    wm, dev, ct = build_model(model_path, dev, ct, allow_cpu_fallback=(device == "auto"))
    logger.info(f"Modèle chargé ({dev}/{ct}) en {time.perf_counter() - t0:.1f}s.")

    stats = {"ok": 0, "skip": 0, "fail": 0}
    for media in media_files:
        try:
            result = process_one(
                wm,
                media,
                lang=lang_arg,
                formats=formats_list,
                out_dir=output_dir,
                beam_size=beam_size,
                vad=vad,
                initial_prompt=initial_prompt,
                condition_previous=condition_previous,
                paragraph_gap=paragraph_gap,
                line_length=line_length,
                use_full_name=use_full_name,
                force=force,
            )
            stats[result] += 1
        except Exception as exc:  # un fichier en échec ne doit pas tuer le lot
            logger.error(f"❌ Échec sur {media.name} : {exc}")
            logger.opt(exception=True).debug("Détail de l'erreur :")
            stats["fail"] += 1

    logger.success(
        f"Terminé : {stats['ok']} transcrit(s), "
        f"{stats['skip']} ignoré(s), {stats['fail']} échec(s)."
    )
    if stats["fail"]:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
