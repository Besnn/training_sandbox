#!/usr/bin/env python3
"""Capture snapshots from an RTSP stream.

Two modes:

  1. Continuous (default): take one snapshot every --interval seconds.
  2. Scheduled: pass --schedule HH:MM (repeatable) or --schedule-file PATH
     to capture only around scheduled times. The script sleeps between
     trains and captures densely during each window. Use this when you
     know roughly when trains pass but they can be early or late.

Saves timestamped JPEGs (or PNGs) into a folder. Uses OpenCV's FFMPEG
backend with TCP transport for reliability and a 1-frame buffer to keep
captured frames close to "now". Reconnects automatically on dropouts.

Usage:
    # continuous
    python3 rtsp_snapshot.py rtsp://user:pass@host:554/stream
    python3 rtsp_snapshot.py <url> --interval 10 --output-dir ./snaps

    # scheduled: ±2 min around each train, one frame per second in-window
    python3 rtsp_snapshot.py <url> \\
        --schedule 08:15 --schedule 12:30 --schedule 18:45 \\
        --window-before 120 --window-after 180 --interval 1

    # schedule file (one HH:MM per line, '#' comments allowed)
    python3 rtsp_snapshot.py <url> --schedule-file trains.txt
"""

import os
# Set BEFORE importing cv2 so the FFMPEG backend picks it up.
# TCP is slower but far more reliable than the default UDP for RTSP.
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import argparse
import datetime as dt
import signal
import time
from pathlib import Path

import cv2


_stop = False


def _handle_signal(signum, frame):
    global _stop
    _stop = True


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


# --------------------------------------------------------------------------- #
# Stream helpers
# --------------------------------------------------------------------------- #
def open_stream(url):
    """Open an RTSP capture or return None on failure."""
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        return None
    # 1-frame buffer keeps grabbed frames close to real time (best effort —
    # not all backends honour this).
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except cv2.error:
        pass
    return cap


def drain_for(cap, seconds):
    """Keep grabbing frames for `seconds` so we don't fall behind the stream.

    Returns False if the stream drops during the wait, True otherwise.
    """
    deadline = time.monotonic() + seconds
    while not _stop and time.monotonic() < deadline:
        if not cap.grab():
            return False
        time.sleep(0.02)
    return True


def save_frame(frame, out_dir, prefix, fmt, encode_params):
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    fname = f"{prefix}_{ts}.{fmt}"
    path = out_dir / fname
    return cv2.imwrite(str(path), frame, encode_params), fname


# --------------------------------------------------------------------------- #
# Schedule parsing
# --------------------------------------------------------------------------- #
DAY_NAMES = {
    "mon": 0, "tue": 1, "wed": 2, "thu": 3, "fri": 4, "sat": 5, "sun": 6,
}


def parse_schedule_time(text):
    """Parse 'HH:MM' or 'HH:MM:SS' into a datetime.time."""
    parts = text.strip().split(":")
    if len(parts) == 2:
        return dt.time(int(parts[0]), int(parts[1]), 0)
    if len(parts) == 3:
        return dt.time(int(parts[0]), int(parts[1]), int(parts[2]))
    raise ValueError(f"Invalid time {text!r}; expected HH:MM or HH:MM:SS")


def load_schedule_file(path):
    times = []
    with open(path) as f:
        for raw in f:
            line = raw.split("#", 1)[0].strip()
            if line:
                times.append(parse_schedule_time(line))
    return times


def parse_days(spec):
    """Parse 'mon,tue,wed' → set({0,1,2}). None → all days allowed."""
    if not spec:
        return None
    out = set()
    for part in spec.split(","):
        key = part.strip().lower()[:3]
        if key not in DAY_NAMES:
            raise ValueError(f"Invalid day {part!r}")
        out.add(DAY_NAMES[key])
    return out


def collect_schedule(args):
    """Combine --schedule and --schedule-file into a sorted, unique list."""
    times = []
    if args.schedule:
        times.extend(parse_schedule_time(s) for s in args.schedule)
    if args.schedule_file:
        times.extend(load_schedule_file(args.schedule_file))
    seen = set()
    out = []
    for t in sorted(times):
        key = (t.hour, t.minute, t.second)
        if key not in seen:
            seen.add(key)
            out.append(t)
    return out


def next_event(now, times, allowed_days, window_after):
    """Find the next (event_dt, time) whose window-end has not passed.

    Looks up to 8 days ahead. Returns None if the schedule yields nothing.
    """
    for day_offset in range(0, 9):
        day = (now + dt.timedelta(days=day_offset)).date()
        if allowed_days is not None and day.weekday() not in allowed_days:
            continue
        day_candidates = []
        for t in times:
            event_dt = dt.datetime.combine(day, t)
            if event_dt + dt.timedelta(seconds=window_after) >= now:
                day_candidates.append((event_dt, t))
        if day_candidates:
            day_candidates.sort(key=lambda x: x[0])
            return day_candidates[0]
    return None


def fmt_duration(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def interruptible_sleep(seconds, status_fn=None, status_every=60.0):
    """Sleep up to `seconds`, waking early on _stop.

    If status_fn is given, calls it with the remaining seconds every
    `status_every` seconds.
    """
    end = time.monotonic() + seconds
    next_status = time.monotonic()
    while not _stop:
        remaining = end - time.monotonic()
        if remaining <= 0:
            return True
        if status_fn and time.monotonic() >= next_status:
            status_fn(remaining)
            next_status = time.monotonic() + status_every
        time.sleep(min(0.5, remaining))
    return False


def _stop_now():
    global _stop
    _stop = True


# --------------------------------------------------------------------------- #
# Capture loops
# --------------------------------------------------------------------------- #
def run_continuous(args, base_out_dir, encode_params):
    """Original behavior: snapshot every --interval forever."""
    print(f"Mode      : continuous, every {args.interval}s")
    print(f"Output    : {base_out_dir.resolve()}\n")

    cap = None
    saved = 0
    while not _stop:
        if cap is None or not cap.isOpened():
            print("Connecting to stream…")
            cap = open_stream(args.rtsp_url)
            if cap is None:
                print(f"  Failed to open. Retry in {args.reconnect_delay}s…")
                time.sleep(args.reconnect_delay)
                continue
            print("  Connected.")

        ok, frame = cap.read()
        if not ok or frame is None:
            print("  Read failed. Reconnecting…")
            cap.release(); cap = None
            time.sleep(args.reconnect_delay)
            continue

        ok_write, fname = save_frame(
            frame, base_out_dir, args.prefix, args.format, encode_params
        )
        if ok_write:
            saved += 1
            print(f"  [{saved:>4}] {fname}  "
                  f"({frame.shape[1]}x{frame.shape[0]})")
        else:
            print(f"  imwrite failed for {fname}")

        if args.max_frames and saved >= args.max_frames:
            print(f"Reached --max-frames={args.max_frames}, stopping.")
            break

        if not drain_for(cap, args.interval):
            print("  Stream dropped during wait. Reconnecting…")
            cap.release(); cap = None
            time.sleep(args.reconnect_delay)

    if cap is not None:
        cap.release()
    print(f"\nDone. Saved {saved} snapshot(s) to {base_out_dir.resolve()}")


def run_scheduled(args, base_out_dir, encode_params, schedule):
    """Wake up around each scheduled time and capture through a window."""
    allowed_days = parse_days(args.days)
    print(f"Mode      : scheduled  "
          f"(window: -{args.window_before:.0f}s / +{args.window_after:.0f}s, "
          f"interval {args.interval}s)")
    print("Schedule  : "
          + ", ".join(t.strftime("%H:%M:%S") for t in schedule))
    if allowed_days is not None:
        day_names = [k for k, v in sorted(DAY_NAMES.items(), key=lambda kv: kv[1])
                     if v in allowed_days]
        print(f"Days      : {','.join(day_names)}")
    print(f"Output    : {base_out_dir.resolve()}")
    print(f"Warmup    : reconnect {args.warmup:.0f}s before window opens\n")

    total_saved = 0
    while not _stop:
        now = dt.datetime.now()
        nxt = next_event(now, schedule, allowed_days, args.window_after)
        if nxt is None:
            print("No upcoming scheduled events. Exiting.")
            break
        event_dt, event_time = nxt
        window_start = event_dt - dt.timedelta(seconds=args.window_before)
        window_end = event_dt + dt.timedelta(seconds=args.window_after)

        # Folder per event so each pass is easy to flip through.
        ev_dir = base_out_dir / event_dt.strftime("%Y%m%d_%H%M")
        ev_prefix = f"{args.prefix}_{event_time.strftime('%H%M')}"

        # ---- Sleep until ~warmup before the window opens ---------------- #
        warmup_at = window_start - dt.timedelta(seconds=args.warmup)
        sleep_secs = (warmup_at - dt.datetime.now()).total_seconds()
        if sleep_secs > 0:
            scheduled_str = event_dt.strftime("%a %Y-%m-%d %H:%M:%S")
            print(f"Next event: {scheduled_str}  "
                  f"(window opens at {window_start.strftime('%H:%M:%S')})")

            def status(remaining):
                eta = fmt_duration(remaining + args.warmup)
                print(f"  waiting… window opens in {eta}")

            if not interruptible_sleep(sleep_secs, status_fn=status,
                                       status_every=60.0):
                break

        if _stop:
            break

        # ---- Open stream & capture through the window ------------------- #
        ev_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n=== Window for {event_dt.strftime('%H:%M:%S')}  "
              f"({window_start.strftime('%H:%M:%S')} → "
              f"{window_end.strftime('%H:%M:%S')}) ===")
        print(f"  Saving to: {ev_dir}")

        cap = None
        win_saved = 0
        while not _stop and dt.datetime.now() < window_end:
            if cap is None or not cap.isOpened():
                print("  Connecting to stream…")
                cap = open_stream(args.rtsp_url)
                if cap is None:
                    print(f"    Failed. Retry in {args.reconnect_delay}s…")
                    if not interruptible_sleep(args.reconnect_delay):
                        break
                    continue
                print("  Connected.")

            # During warmup, just drain the stream — don't save yet.
            if dt.datetime.now() < window_start:
                remaining = (window_start - dt.datetime.now()).total_seconds()
                if not drain_for(cap, min(remaining, 1.0)):
                    print("  Stream dropped during warmup. Reconnecting…")
                    cap.release(); cap = None
                continue

            ok, frame = cap.read()
            if not ok or frame is None:
                print("  Read failed. Reconnecting…")
                cap.release(); cap = None
                if not interruptible_sleep(args.reconnect_delay):
                    break
                continue

            ok_write, fname = save_frame(
                frame, ev_dir, ev_prefix, args.format, encode_params
            )
            if ok_write:
                win_saved += 1
                total_saved += 1
                print(f"    [{win_saved:>4}] {fname}  "
                      f"({frame.shape[1]}x{frame.shape[0]})")
            else:
                print(f"    imwrite failed for {fname}")

            if args.max_frames and total_saved >= args.max_frames:
                print(f"Reached --max-frames={args.max_frames}, stopping.")
                _stop_now()
                break

            # Drain until next snapshot OR window end, whichever comes first.
            time_to_window_end = (window_end - dt.datetime.now()).total_seconds()
            wait = min(args.interval, max(0.0, time_to_window_end))
            if wait > 0 and not drain_for(cap, wait):
                print("  Stream dropped during wait. Reconnecting…")
                cap.release(); cap = None

        if cap is not None:
            cap.release()
        print(f"=== Window closed. {win_saved} snapshot(s) saved. ===")

    print(f"\nDone. Saved {total_saved} snapshot(s) total to "
          f"{base_out_dir.resolve()}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("rtsp_url",
                    help="RTSP URL, e.g. rtsp://user:pass@192.168.1.10:554/stream1")
    ap.add_argument("--output-dir", default=None,
                    help="Folder for snapshots. Default: ./snapshots/<timestamp>")
    ap.add_argument("--interval", type=float, default=5.0,
                    help="Seconds between snapshots. In scheduled mode this is "
                         "the in-window interval (default: %(default)s; for "
                         "train passes try 0.5-1.0)")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Stop after this many snapshots total "
                         "(default: 0 = unlimited)")
    ap.add_argument("--prefix", default="snap",
                    help="Filename prefix (default: %(default)s)")
    ap.add_argument("--format", default="jpg", choices=["jpg", "png"],
                    help="Output image format (default: %(default)s)")
    ap.add_argument("--jpg-quality", type=int, default=95,
                    help="JPEG quality 1-100 (default: %(default)s)")
    ap.add_argument("--reconnect-delay", type=float, default=2.0,
                    help="Seconds to wait before reconnecting after a failure "
                         "(default: %(default)s)")
    ap.add_argument("--transport", choices=["tcp", "udp"], default="tcp",
                    help="RTSP transport (default: %(default)s)")

    # Scheduling
    ap.add_argument("--schedule", action="append", default=None,
                    metavar="HH:MM",
                    help="Scheduled train time (repeatable). Enables "
                         "scheduled mode.")
    ap.add_argument("--schedule-file", default=None,
                    help="Text file with one HH:MM[:SS] per line. '#' starts a "
                         "comment.")
    ap.add_argument("--window-before", type=float, default=120.0,
                    help="Seconds before the scheduled time to start capturing "
                         "(default: %(default)s)")
    ap.add_argument("--window-after", type=float, default=180.0,
                    help="Seconds after the scheduled time to keep capturing "
                         "(default: %(default)s)")
    ap.add_argument("--warmup", type=float, default=15.0,
                    help="Seconds before window-start to (re)open the stream "
                         "so the first snapshot is ready on time "
                         "(default: %(default)s)")
    ap.add_argument("--days", default=None,
                    help="Comma-separated weekdays to honour, e.g. "
                         "'mon,tue,wed,thu,fri'. Omitted = every day.")
    return ap.parse_args()


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    args = parse_args()

    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = f"rtsp_transport;{args.transport}"

    if args.output_dir is None:
        run_tag = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_out_dir = Path("snapshots") / run_tag
    else:
        base_out_dir = Path(args.output_dir)
    base_out_dir.mkdir(parents=True, exist_ok=True)

    encode_params = []
    if args.format == "jpg":
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(args.jpg_quality)]

    print(f"RTSP URL  : {args.rtsp_url}")
    print(f"Transport : {args.transport}")

    schedule = collect_schedule(args)
    print("Press Ctrl+C to stop.\n")

    if schedule:
        run_scheduled(args, base_out_dir, encode_params, schedule)
    else:
        run_continuous(args, base_out_dir, encode_params)


if __name__ == "__main__":
    main()
