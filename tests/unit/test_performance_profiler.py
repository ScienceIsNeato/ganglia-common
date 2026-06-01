from unittest.mock import patch

from ganglia_common.logger import Logger
from ganglia_common.utils import performance_profiler as profiler


def test_performance_stats_record_summarize_and_reset():
    stats = profiler.PerformanceStats()
    assert stats.get_stats("missing") is None

    stats.record("LLM", 1.0)
    stats.record("LLM", 3.0)
    summary = stats.get_stats("LLM")

    assert summary == {
        "count": 2,
        "mean": 2.0,
        "median": 2.0,
        "p95": 3.0,
        "p99": 3.0,
        "min": 1.0,
        "max": 3.0,
    }
    with patch.object(Logger, "print_info") as print_info:
        stats.print_summary()
    assert any("PERFORMANCE SUMMARY" in str(call) for call in print_info.call_args_list)

    stats.reset()
    assert stats.timings == {}


def test_timer_and_timed_decorator_collect_stats():
    profiler.disable_timing_analysis()
    assert profiler.is_timing_enabled() is False
    with profiler.Timer("disabled"):
        pass
    assert "disabled" not in profiler.get_global_stats().timings

    profiler.enable_timing_analysis()
    profiler.get_global_stats().reset()
    with (
        patch.object(profiler.time, "time", side_effect=[1.0, 1.5, 2.0, 2.25]),
        patch.object(Logger, "print_perf"),
        patch.object(Logger, "print_info"),
    ):
        with profiler.Timer("context"):
            pass

        @profiler.timed(name="decorated")
        def operation():
            return "done"

        assert operation() == "done"

    assert profiler.get_global_stats().timings == {
        "context": [0.5],
        "decorated": [0.25],
    }
    profiler.disable_timing_analysis()


def test_conversation_timer_records_durations_and_breakdown():
    timer = profiler.ConversationTimer()
    assert timer.get_stt_duration() is None
    assert timer.get_llm_duration() is None
    assert timer.get_tts_duration() is None
    assert timer.get_roundtrip_duration() is None
    assert timer.get_user_duration() is None
    assert timer.get_ai_duration() is None

    with patch.object(profiler.time, "time", side_effect=range(1, 12)):
        timer.mark_user_start()
        timer.mark_user_end()
        timer.mark_stt_start()
        timer.mark_stt_end()
        timer.mark_ai_start()
        timer.mark_llm_start()
        timer.mark_llm_end()
        timer.mark_tts_start()
        timer.mark_tts_end()
        timer.mark_playback_start()
        timer.mark_ai_end()

    assert timer.get_user_duration() == 1
    assert timer.get_stt_duration() == 1
    assert timer.get_llm_duration() == 1
    assert timer.get_tts_duration() == 1
    assert timer.get_roundtrip_duration() == 8
    assert timer.get_ai_duration() == 6

    profiler.enable_timing_analysis()
    profiler.get_global_stats().reset()
    with patch.object(Logger, "print_perf") as print_perf:
        timer.print_breakdown()
    assert any("TOTAL LATENCY" in str(call) for call in print_perf.call_args_list)
    assert profiler.get_global_stats().timings["Roundtrip"] == [8]
    profiler.disable_timing_analysis()


def test_conversation_breakdown_requires_timing_data():
    timer = profiler.ConversationTimer()
    profiler.enable_timing_analysis()
    with patch.object(Logger, "print_perf") as print_perf:
        timer.print_breakdown()
    assert any("No timing data" in str(call) for call in print_perf.call_args_list)
    profiler.disable_timing_analysis()
