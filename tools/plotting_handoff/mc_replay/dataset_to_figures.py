"""Convert deployment/raw daily files into selectable four-panel PNG figures."""

from __future__ import annotations

import argparse
from datetime import date, datetime, time
from pathlib import Path

from common_4panel_plotting import BUNDLE_DIR, OUTPUT_DIR, load_data
from outdata_folder_loader import load_outdata_directory
from plot_4panel_mc_command import save_command_figure
from plot_4panel_voltage_current import save_voltage_current_figure


DEFAULT_OUTDATA_DIR = BUNDLE_DIR / "dataset"


def _date_value(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Use YYYY-MM-DD") from exc


def _time_value(value: str) -> time:
    for pattern in ("%H:%M", "%H:%M:%S"):
        try:
            return datetime.strptime(value, pattern).time()
        except ValueError:
            continue
    raise argparse.ArgumentTypeError("Use HH:MM or HH:MM:SS")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create four-panel figures from deployment_v2/raw_data_v2 files."
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--data-dir",
        type=Path,
        help="Folder containing deployment_v2_*.csv and raw_data_v2_*.csv",
    )
    source.add_argument(
        "--input-csv",
        "--input",
        dest="input_csv",
        type=Path,
        help="Optional pre-converted single CSV input",
    )
    parser.add_argument("--start-date", type=_date_value, help="First date, YYYY-MM-DD")
    parser.add_argument("--end-date", type=_date_value, help="Last date, YYYY-MM-DD")
    parser.add_argument(
        "--start-time",
        type=_time_value,
        default=time(0, 0),
        help="Time on the first date, HH:MM (default: 00:00)",
    )
    parser.add_argument(
        "--end-time",
        type=_time_value,
        default=time(23, 59, 59),
        help="Time on the last date, HH:MM (default: 23:59:59)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory for generated PNG files",
    )
    parser.add_argument(
        "--view",
        choices=("both", "command", "voltage-current"),
        default="both",
        help="Select the fourth-panel content",
    )
    parser.add_argument("--name", help="Output filename prefix")
    parser.add_argument("--title-prefix", help="Main title prefix")
    return parser.parse_args()


def _default_name(frame, args: argparse.Namespace) -> str:
    first = frame["timestamp"].min()
    last = frame["timestamp"].max()
    date_part = f"{first:%Y%m%d}_{last:%Y%m%d}"
    time_part = f"{args.start_time:%H%M}_{args.end_time:%H%M}"
    return f"{date_part}_{time_part}"


def main() -> None:
    args = _parse_args()
    if args.input_csv is not None:
        if args.start_date is not None or args.end_date is not None:
            raise SystemExit("--start-date/--end-date require --data-dir")
        frame = load_data(args.input_csv)
        source_description = str(args.input_csv.expanduser().resolve())
    else:
        data_dir = args.data_dir if args.data_dir is not None else DEFAULT_OUTDATA_DIR
        frame = load_outdata_directory(
            data_dir,
            start_date=args.start_date,
            end_date=args.end_date,
            start_time=args.start_time,
            end_time=args.end_time,
        )
        source_description = str(data_dir.expanduser().resolve())

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    name = args.name or _default_name(frame, args)
    title_prefix = args.title_prefix or (
        f"{frame['timestamp'].min():%Y-%m-%d %H:%M} to "
        f"{frame['timestamp'].max():%Y-%m-%d %H:%M}"
    )

    outputs: list[Path] = []
    if args.view in {"both", "command"}:
        outputs.append(
            save_command_figure(
                frame,
                output_dir / f"{name}_mc_command.png",
                f"{title_prefix} — command view",
            )
        )
    if args.view in {"both", "voltage-current"}:
        outputs.append(
            save_voltage_current_figure(
                frame,
                output_dir / f"{name}_voltage_current.png",
                f"{title_prefix} — voltage/current view",
            )
        )

    print(f"Source: {source_description}")
    print(
        f"Selected: {frame['timestamp'].min()} to {frame['timestamp'].max()} "
        f"({len(frame)} deployment rows)"
    )
    for output in outputs:
        print(f"Saved: {output}")


if __name__ == "__main__":
    main()
