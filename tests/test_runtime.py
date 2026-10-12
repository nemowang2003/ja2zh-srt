import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from audio_transcribe import Segment, Transcript
from click.testing import CliRunner
from ja2zh_srt import PipelineOptions, main, process_target
from ja2zh_srt.subtitles import SubtitleSegment
from ja2zh_srt.translate import (
    SakuraConfig,
    TranslationError,
    parse_translation_response,
    translate_segments,
)


class PipelineTests(unittest.TestCase):
    def options(self):
        return PipelineOptions(
            False,
            "large-v3",
            "cuda",
            None,
            5,
            True,
            8,
            0.3,
            0.8,
            "http://localhost:4000/v1",
            "test-key",
            "sakura",
        )

    def test_sdk_transcription_then_api_translation_and_skip(self):
        with tempfile.TemporaryDirectory() as temp:
            audio = Path(temp) / "input.mp3"
            audio.touch()
            result = Transcript((Segment(1, 2, "日本語"),), "ja")
            with (
                patch("ja2zh_srt.Transcriber") as engine,
                patch("ja2zh_srt.translate_file") as translate,
            ):
                engine.return_value.__enter__.return_value.transcribe.return_value = result
                process_target(audio, self.options())
                self.assertIn(
                    "00:00:01,000 --> 00:00:02,000", audio.with_suffix(".ja.srt").read_text()
                )
                engine.assert_called_once_with(model="large-v3", device="cuda", compute_type=None)
                self.assertEqual(translate.call_args.kwargs["config"].model, "sakura")
                audio.with_suffix(".zh.srt").touch()
                process_target(audio, self.options())
                self.assertEqual(engine.call_count, 1)
                self.assertEqual(translate.call_count, 1)

    def test_srt_translation_does_not_load_whisper(self):
        with tempfile.TemporaryDirectory() as temp:
            subtitle = Path(temp) / "input.ja.srt"
            subtitle.write_text("1\n00:00:00,000 --> 00:00:01,000\n日本語\n")
            with patch("ja2zh_srt.Transcriber") as engine, patch("ja2zh_srt.translate_file"):
                process_target(subtitle, self.options())
                engine.assert_not_called()

    def test_cuda_check_failure_is_actionable(self):
        with patch(
            "ja2zh_srt.compute_type_for", side_effect=RuntimeError("no usable acceleration found")
        ):
            result = CliRunner().invoke(main, ["check"])
            self.assertNotEqual(result.exit_code, 0)
            self.assertIn("no usable acceleration", result.output)

    def test_translation_preserves_cue_identity_and_times(self):
        segments = [SubtitleSegment(7, 1.25, 2.5, "こんにちは")]
        output = translate_segments(
            segments,
            config=SakuraConfig("http://test", "key", "sakura"),
            complete=lambda _: "[1]\n你好",
        )
        self.assertEqual(
            (output[0].index, output[0].start, output[0].end, output[0].text),
            (7, 1.25, 2.5, "你好"),
        )

    def test_missing_and_duplicate_ids_rejected(self):
        for text in ["[1]\n你好", "[1]\n你好\n[1]\n再见"]:
            with self.assertRaises(TranslationError):
                parse_translation_response(text, expected_ids=[1, 2])
