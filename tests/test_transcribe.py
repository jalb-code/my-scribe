"""Tests unitaires des fonctions pures de transcribe.py (sans chargement de modèle)."""

from __future__ import annotations

import json
from pathlib import Path

import transcribe as t


# ---------------------------------------------------------------------------
# fmt_ts : formatage des horodatages
# ---------------------------------------------------------------------------
def test_fmt_ts_zero():
    assert t.fmt_ts(0) == "00:00:00.000"


def test_fmt_ts_heures_minutes_secondes_ms():
    # 1 h 1 min 1 s 500 ms
    assert t.fmt_ts(3661.5) == "01:01:01.500"


def test_fmt_ts_separateur_srt():
    assert t.fmt_ts(1.234, ",") == "00:00:01,234"


def test_fmt_ts_arrondi_ms():
    # 0.0005 s -> 1 ms (arrondi)
    assert t.fmt_ts(0.0005).endswith(".001") or t.fmt_ts(0.0005).endswith(".000")


# ---------------------------------------------------------------------------
# group_paragraphs : regroupement en paragraphes selon la pause
# ---------------------------------------------------------------------------
def _seg(start, end, text):
    return {"start": start, "end": end, "text": text}


def test_group_paragraphs_pause_courte_un_seul_paragraphe():
    segments = [_seg(0.0, 1.0, "Bonjour"), _seg(1.2, 2.0, "tout le monde")]
    # gap 1.5 > 0.2 -> pas de coupure
    assert t.group_paragraphs(segments, gap=1.5, line_length=0) == "Bonjour tout le monde"


def test_group_paragraphs_pause_longue_deux_paragraphes():
    segments = [_seg(0.0, 1.0, "Premier"), _seg(5.0, 6.0, "Second")]
    out = t.group_paragraphs(segments, gap=1.5, line_length=0)
    assert out == "Premier\n\nSecond"


def test_group_paragraphs_wrap_line_length():
    segments = [_seg(0.0, 1.0, "mot " * 20)]
    out = t.group_paragraphs(segments, gap=1.5, line_length=20)
    assert all(len(line) <= 20 for line in out.splitlines())


# ---------------------------------------------------------------------------
# render_vtt / render_srt
# ---------------------------------------------------------------------------
def test_render_vtt_entete_et_horodatage():
    segments = [_seg(0.0, 1.5, "Salut")]
    out = t.render_vtt(segments)
    assert out.startswith("WEBVTT")
    assert "00:00:00.000 --> 00:00:01.500" in out
    assert "Salut" in out


def test_render_srt_index_et_virgule():
    segments = [_seg(0.0, 1.5, "Salut"), _seg(2.0, 3.0, "Ça va")]
    out = t.render_srt(segments)
    lines = out.splitlines()
    assert lines[0] == "1"
    assert lines[1] == "00:00:00,000 --> 00:00:01,500"
    assert "2" in lines  # deuxième segment indexé


# ---------------------------------------------------------------------------
# render : dispatch selon le format, dont json
# ---------------------------------------------------------------------------
def test_render_json_structure():
    segments = [_seg(0.0, 1.0, "Test")]
    info = {"language": "fr", "language_probability": 0.99, "duration": 1.0}
    payload = json.loads(t.render("json", segments, info, paragraph_gap=1.5, line_length=0))
    assert payload["language"] == "fr"
    assert payload["duration"] == 1.0
    assert payload["segments"] == segments


def test_render_format_inconnu_leve():
    import pytest

    with pytest.raises(ValueError):
        t.render("xml", [], {}, paragraph_gap=1.5, line_length=0)


# ---------------------------------------------------------------------------
# discover_media : découverte des fichiers média
# ---------------------------------------------------------------------------
def test_discover_media_fichier_unique(tmp_path: Path):
    f = tmp_path / "audio.mp3"
    f.write_bytes(b"")
    assert t.discover_media(f) == [f]


def test_discover_media_filtre_et_trie(tmp_path: Path):
    (tmp_path / "b.mp3").write_bytes(b"")
    (tmp_path / "a.wav").write_bytes(b"")
    (tmp_path / "notes.txt").write_bytes(b"")  # ignoré (pas un média)
    (tmp_path / "clip.mp4").write_bytes(b"")  # vidéo prise en charge
    found = [p.name for p in t.discover_media(tmp_path)]
    assert found == ["a.wav", "b.mp3", "clip.mp4"]


# ---------------------------------------------------------------------------
# build_output_path
# ---------------------------------------------------------------------------
def test_build_output_path_stem_simple(tmp_path: Path):
    media = Path("reunion.m4a")
    out = t.build_output_path(media, tmp_path, "txt", use_full_name=False)
    assert out == tmp_path / "reunion.txt"


def test_build_output_path_nom_complet_anticollision(tmp_path: Path):
    media = Path("reunion.m4a")
    out = t.build_output_path(media, tmp_path, "vtt", use_full_name=True)
    assert out == tmp_path / "reunion.m4a.vtt"


# ---------------------------------------------------------------------------
# resolve_device / resolve_compute_type
# ---------------------------------------------------------------------------
def test_resolve_device_explicite_inchange():
    assert t.resolve_device("cpu") == "cpu"


def test_resolve_compute_type_cpu_int8():
    assert t.resolve_compute_type("auto", "cpu") == "int8"


def test_resolve_compute_type_cuda_float16():
    assert t.resolve_compute_type("auto", "cuda") == "float16"


def test_resolve_compute_type_explicite_inchange():
    assert t.resolve_compute_type("float32", "cpu") == "float32"
