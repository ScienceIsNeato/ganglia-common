"""Performance profiling utilities for GANGLIA.

This module provides timing instrumentation to measure conversation pipeline performance
and identify bottlenecks in the speech-to-text, AI query, and text-to-speech stages.
"""

import time
import functools
from typing import Optional, Dict, List
from contextlib import contextmanager
from ganglia_common.logger import Logger


# Global flag to control timing analysis
_timing_enabled = False


def enable_timing_analysis():
    """Enable timing analysis globally."""
    global _timing_enabled
    _timing_enabled = True


def disable_timing_analysis():
    """Disable timing analysis globally."""
    global _timing_enabled
    _timing_enabled = False


def is_timing_enabled() -> bool:
    """Check if timing analysis is enabled."""
    return _timing_enabled


class PerformanceStats:
    """Collect and analyze performance statistics."""

    def __init__(self):
        self.timings: Dict[str, List[float]] = {}

    def record(self, name: str, duration: float):
        """Record a timing measurement.

        Args:
            name: Name of the measured operation
            duration: Duration in seconds
        """
        if name not in self.timings:
            self.timings[name] = []
        self.timings[name].append(duration)

    def get_stats(self, name: str) -> Optional[Dict[str, float]]:
        """Get statistics for a named operation.

        Args:
            name: Name of the operation

        Returns:
            Dictionary with mean, median, p95, p99, min, max stats
        """
        if name not in self.timings or not self.timings[name]:
            return None

        values = sorted(self.timings[name])
        n = len(values)

        # Calculate median correctly for even and odd datasets
        if n % 2 == 0:
            median = (values[n // 2 - 1] + values[n // 2]) / 2
        else:
            median = values[n // 2]

        return {
            "count": n,
            "mean": sum(values) / n,
            "median": median,
            "p95": values[int(n * 0.95)] if n > 1 else values[0],
            "p99": values[int(n * 0.99)] if n > 1 else values[0],
            "min": values[0],
            "max": values[-1],
        }

    def print_summary(self):
        """Print a summary of all collected statistics."""
        Logger.print_info("=" * 60)
        Logger.print_info("PERFORMANCE SUMMARY")
        Logger.print_info("=" * 60)

        for name in sorted(self.timings.keys()):
            stats = self.get_stats(name)
            if stats:
                Logger.print_info(f"\n{name}:")
                Logger.print_info(f"  Count:   {stats['count']}")
                Logger.print_info(f"  Mean:    {stats['mean']:.2f}s")
                Logger.print_info(f"  Median:  {stats['median']:.2f}s")
                Logger.print_info(f"  P95:     {stats['p95']:.2f}s")
                Logger.print_info(f"  P99:     {stats['p99']:.2f}s")
                Logger.print_info(f"  Min:     {stats['min']:.2f}s")
                Logger.print_info(f"  Max:     {stats['max']:.2f}s")

        Logger.print_info("=" * 60)

    def reset(self):
        """Clear all collected statistics."""
        self.timings.clear()


# Global performance stats instance
_global_stats = PerformanceStats()


def get_global_stats() -> PerformanceStats:
    """Get the global performance statistics instance."""
    return _global_stats


@contextmanager
def Timer(name: str, log: bool = True, collect_stats: bool = True):
    """Context manager for timing code blocks.

    Args:
        name: Name of the operation being timed
        log: Whether to log the timing immediately
        collect_stats: Whether to collect stats for later analysis

    Usage:
        with Timer("my_operation"):
            do_something()
    """
    if not _timing_enabled:
        yield
        return

    start = time.time()
    yield
    elapsed = time.time() - start

    if log:
        Logger.print_perf(f"⏱️  {name}: {elapsed:.2f}s")

    if collect_stats:
        _global_stats.record(name, elapsed)


def timed(name: Optional[str] = None, log: bool = True, collect_stats: bool = True):
    """Decorator for timing function execution.

    Args:
        name: Custom name for the operation (defaults to function name)
        log: Whether to log the timing immediately
        collect_stats: Whether to collect stats for later analysis

    Usage:
        @timed()
        def my_function():
            pass

        @timed(name="Custom Operation")
        def another_function():
            pass
    """

    def decorator(func):
        operation_name = name or func.__name__

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                elapsed = time.time() - start

                if log:
                    Logger.print_info(f"⏱️  {operation_name}: {elapsed:.2f}s")

                if collect_stats:
                    _global_stats.record(operation_name, elapsed)

        return wrapper

    return decorator


class ConversationTimer:
    """Track timing for a complete conversation turn."""

    def __init__(self):
        self.user_start = None
        self.user_end = None
        self.ai_start = None
        self.ai_end = None
        self.stt_start = None
        self.stt_end = None
        self.llm_start = None
        self.llm_end = None
        self.tts_start = None
        self.tts_end = None
        self.playback_start = None

    def mark_user_start(self):
        """Mark start of user turn."""
        self.user_start = time.time()

    def mark_user_end(self):
        """Mark end of user turn (finished speaking)."""
        self.user_end = time.time()

    def mark_stt_start(self):
        """Mark start of speech-to-text processing."""
        self.stt_start = time.time()

    def mark_stt_end(self):
        """Mark end of speech-to-text processing."""
        self.stt_end = time.time()

    def mark_ai_start(self):
        """Mark start of AI turn."""
        self.ai_start = time.time()

    def mark_llm_start(self):
        """Mark start of LLM query."""
        self.llm_start = time.time()

    def mark_llm_end(self):
        """Mark end of LLM query."""
        self.llm_end = time.time()

    def mark_tts_start(self):
        """Mark start of text-to-speech generation."""
        self.tts_start = time.time()

    def mark_tts_end(self):
        """Mark end of text-to-speech generation."""
        self.tts_end = time.time()

    def mark_playback_start(self):
        """Mark start of audio playback."""
        self.playback_start = time.time()

    def mark_ai_end(self):
        """Mark end of AI turn."""
        self.ai_end = time.time()

    def get_stt_duration(self) -> Optional[float]:
        """Get STT duration in seconds."""
        if self.stt_start and self.stt_end:
            return self.stt_end - self.stt_start
        return None

    def get_llm_duration(self) -> Optional[float]:
        """Get LLM query duration in seconds."""
        if self.llm_start and self.llm_end:
            return self.llm_end - self.llm_start
        return None

    def get_tts_duration(self) -> Optional[float]:
        """Get TTS duration in seconds."""
        if self.tts_start and self.tts_end:
            return self.tts_end - self.tts_start
        return None

    def get_roundtrip_duration(self) -> Optional[float]:
        """Get total roundtrip duration from user stops speaking to AI starts speaking.

        This is the key metric: time from end of user speech to start of AI audio playback.
        """
        if self.user_end and self.playback_start:
            return self.playback_start - self.user_end
        return None

    def get_user_duration(self) -> Optional[float]:
        """Get user turn duration in seconds."""
        if self.user_start and self.user_end:
            return self.user_end - self.user_start
        return None

    def get_ai_duration(self) -> Optional[float]:
        """Get AI turn duration in seconds."""
        if self.ai_start and self.ai_end:
            return self.ai_end - self.ai_start
        return None

    def print_breakdown(self):
        """Print visual timeline from user stops speaking to AI starts speaking."""
        if not _timing_enabled:
            return

        # Use user_end as T=0 (moment user stopped speaking)
        if not self.user_end:
            Logger.print_perf("⚠️  No timing data available (user_end not set)")
            return

        t0 = self.user_end

        Logger.print_perf("")
        Logger.print_perf("=" * 80)
        Logger.print_perf("🎯 RESPONSE LATENCY TIMELINE (T=0 = User Stopped Speaking)")
        Logger.print_perf("=" * 80)

        timeline = []

        # STT finalization (silence detection already happened)
        if self.stt_end:
            elapsed = self.stt_end - t0
            delta = self.stt_end - t0  # First event, so delta = elapsed
            timeline.append(("STT Finalized", elapsed, delta))

        # LLM query start
        if self.llm_start:
            elapsed = self.llm_start - t0
            prev = self.stt_end if self.stt_end else t0
            delta = self.llm_start - prev
            timeline.append(("LLM Query Started", elapsed, delta))

        # LLM first response (TTFB - when streaming begins)
        if self.llm_end:
            elapsed = self.llm_end - t0
            prev = self.llm_start if self.llm_start else t0
            delta = self.llm_end - prev
            timeline.append(("LLM First Sentence Ready", elapsed, delta))

        # TTS for first sentence complete
        if self.tts_start and self.tts_end:
            # For streaming, tts_start/end are marked together
            elapsed = self.tts_end - t0
            prev = self.llm_end if self.llm_end else t0
            delta = self.tts_end - prev
            timeline.append(("First TTS Generated", elapsed, delta))

        # Audio playback starts (THE GOAL!)
        if self.playback_start:
            elapsed = self.playback_start - t0
            prev = self.tts_end if self.tts_end else t0
            delta = self.playback_start - prev
            timeline.append(("🔊 AUDIO PLAYBACK START", elapsed, delta))

        # Print timeline
        for event, elapsed, delta in timeline:
            # Format: Event name | T+elapsed | (Δ delta)
            event_str = f"{event:<30}"
            elapsed_str = f"T+{elapsed:>5.2f}s"
            delta_str = f"(Δ {delta:>5.2f}s)"

            if "AUDIO PLAYBACK" in event:
                Logger.print_perf(f"  {event_str} {elapsed_str}  {delta_str}  🎃")
            else:
                Logger.print_perf(f"  {event_str} {elapsed_str}  {delta_str}")

        # Print total roundtrip
        roundtrip = self.get_roundtrip_duration()
        if roundtrip:
            Logger.print_perf("")
            Logger.print_perf(
                f"⏱️  TOTAL LATENCY: {roundtrip:.2f}s (user stops speaking → AI audio starts)"
            )

        Logger.print_perf("=" * 80)
        Logger.print_perf("")

        # Record stats
        stt_dur = self.get_stt_duration()
        llm_dur = self.get_llm_duration()
        tts_dur = self.get_tts_duration()

        if stt_dur:
            _global_stats.record("STT", stt_dur)
        if llm_dur:
            _global_stats.record("LLM", llm_dur)
        if tts_dur:
            _global_stats.record("TTS", tts_dur)
        if roundtrip:
            _global_stats.record("Roundtrip", roundtrip)
