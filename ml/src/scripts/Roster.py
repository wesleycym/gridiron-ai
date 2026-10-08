'''

Usage:
    python ml/src/scripts/roster.py
    python ml/src/scripts/roster.py --slots QB,RB,RB,WR,WR,WR,TE,FLEX,FLEX   # your league's lineup
    python ml/src/scripts/roster.py --file ml/other_team.txt
    python ml/src/scripts/roster.py --season 2025 --week 10 --show-actual  # backtest a past week

'''
import argparse
import re
import sys

import polars as pl

from build_dataset import ML_ROOT, clean, load_raw, load_schedules
from player import injury_line
from predict import next_unplayed_week, predict_week

DEFAULT_FILE = ML_ROOT / "roster.txt"
DEFAULT_SLOTS = "QB,RB,RB,WR,WR,TE,FLEX"
FLEX_OK = {"FLEX": {"RB", "WR", "TE"}, "SUPERFLEX": {"QB", "RB", "WR", "TE"}}
CLOSE_CALL = 1.5  # points; bench players within this of a starter get flagged
DEFENSE_WORDS = {"d", "dst", "def", "defense", "d/st"}


def norm(name: str) -> str:
    """'Amon Ra St Brown' and 'Amon-Ra St. Brown' both become 'amonrastbrown'."""
    name = re.sub(r"\b(jr|sr|ii|iii|iv)\b\.?", "", name.lower())
    return re.sub(r"[^a-z0-9]", "", name)


def read_roster(path) -> list[str]:
    if not path.exists():
        sys.exit(f"No roster file at {path}. Create it with one player per line.")
    lines = [l.strip() for l in path.read_text(encoding="utf-8").splitlines()]
    return [l for l in lines if l and not l.startswith("#")]


def match_players(names: list[str], raw: pl.DataFrame):
    """Map each roster name to a player_id. Returns (matched, notes)."""
    people = (
        raw.sort(["season", "week"])
        .group_by("player_id")
        .agg(pl.col("player_display_name").last(), pl.col("position").last(),
             pl.col("team").last(), pl.col("season").last().alias("last_season"))
        .with_columns(pl.col("player_display_name").map_elements(norm, return_dtype=pl.Utf8).alias("key"))
    )
    matched, notes = [], []
    for name in names:
        words = name.lower().replace(".", "").split()
        if words and words[-1] in DEFENSE_WORDS:
            notes.append(f"{name}: team defense, not projected by the model")
            continue
        key = norm(name)
        hits = people.filter(pl.col("key") == key)
        if hits.is_empty():  # fall back to partial match
            hits = people.filter(pl.col("key").str.contains(key, literal=True))
        if hits.is_empty():
            notes.append(f"{name}: no player found with that name")
            continue
        p = hits.sort("last_season", descending=True).row(0, named=True)
        if p["position"] not in ("QB", "RB", "WR", "TE"):
            notes.append(f"{p['player_display_name']}: {p['position']}, not projected by the model")
            continue
        matched.append(p)
    return matched, notes


def pick_lineup(team: pl.DataFrame, slots: list[str]):
    """Greedy: fill fixed positions with the highest means first, then flex spots."""
    pool = team.filter(pl.col("mean").is_not_null()).sort("mean", descending=True).to_dicts()
    used, lineup = set(), []
    fixed = [s for s in slots if s not in FLEX_OK]
    flex = [s for s in slots if s in FLEX_OK]
    for slot in fixed + flex:
        allowed = FLEX_OK.get(slot, {slot})
        best = next((p for p in pool if p["player_id"] not in used and p["position"] in allowed), None)
        if best:
            used.add(best["player_id"])
        lineup.append((slot, best))
    bench = [p for p in pool if p["player_id"] not in used]
    return lineup, bench


def main() -> None:
    parser = argparse.ArgumentParser(description="Projections and lineup for your roster.")
    parser.add_argument("--file", default=str(DEFAULT_FILE))
    parser.add_argument("--slots", default=DEFAULT_SLOTS,
                        help=f"comma-separated lineup slots (default {DEFAULT_SLOTS}); FLEX = RB/WR/TE")
    parser.add_argument("--season", type=int)
    parser.add_argument("--week", type=int)
    parser.add_argument("--show-actual", action="store_true")
    args = parser.parse_args()

    from pathlib import Path
    names = read_roster(Path(args.file))
    raw = load_raw().filter(pl.col("season_type") == "REG")
    players, notes = match_players(names, raw)

    schedules = load_schedules()
    if args.season and args.week:
        season, week = args.season, args.week
    else:
        season, week = next_unplayed_week(schedules)

    games = clean(raw)
    preds = predict_week(season, week, schedules, games)
    ids = [p["player_id"] for p in players]

    # Position ranks for context (e.g. WR12)
    ranked = preds.with_columns(
        pl.col("mean").rank("ordinal", descending=True).over("position").cast(pl.Int64).alias("pos_rank")
    )
    team = pl.DataFrame(players).select("player_id", "player_display_name", "position", "team").join(
        ranked.select("player_id", "opponent_team", "vegas_is_home", "mean", "median", "floor",
                      "ceiling", "p_15plus", "p_20plus", "pos_rank"),
        on="player_id", how="left",
    )
    if args.show_actual:
        team = team.join(
            games.filter((pl.col("season") == season) & (pl.col("week") == week))
            .select("player_id", pl.col("fantasy_points_ppr").round(1).alias("actual")),
            on="player_id", how="left",
        )

    # ---- Full roster table ----------------------------------------------------------------
    order = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
    rows = []
    for r in sorted(team.to_dicts(), key=lambda r: (order[r["position"]], -(r["mean"] or -1))):
        has = r["mean"] is not None
        opp = (("vs " if r["vegas_is_home"] == 1 else "@ ") + r["opponent_team"]) if has else "BYE/none"
        inj = injury_line(r["player_id"], season, week)
        row = {
            "pos": r["position"],
            "player": r["player_display_name"],
            "rank": f"{r['position']}{r['pos_rank']}" if has else "-",
            "opp": opp,
            "mean": r["mean"], "median": r["median"], "floor": r["floor"], "ceiling": r["ceiling"],
            "15+%": round(r["p_15plus"] * 100) if has else None,
            "20+%": round(r["p_20plus"] * 100) if has else None,
            "injury": "-" if inj == "not on the injury report" else inj[:38],
        }
        if args.show_actual:
            row["actual"] = r.get("actual")
        rows.append(row)

    print(f"\nYOUR ROSTER: WEEK {week}, {season}")
    with pl.Config(tbl_rows=40, tbl_cols=20, tbl_width_chars=160, fmt_str_lengths=40,
                   tbl_hide_dataframe_shape=True, tbl_hide_column_data_types=True):
        print(pl.DataFrame(rows))

    # ---- Suggested lineup -------------------------------------------------------------------
    slots = [s.strip().upper() for s in args.slots.split(",") if s.strip()]
    lineup, bench = pick_lineup(team, slots)
    print(f"\nSUGGESTED LINEUP (highest projected mean; slots: {', '.join(slots)})")
    total = 0.0
    for slot, p in lineup:
        if p is None:
            print(f"  {slot:<5} (nobody available)")
            continue
        total += p["mean"]
        print(f"  {slot:<5} {p['player_display_name']:<24} {p['mean']:5.1f}  "
              f"(range {p['floor']:.1f}-{p['ceiling']:.1f})")
    print(f"  {'':<5} {'Projected total':<24} {total:5.1f}")
    if bench:
        print("\nBENCH")
        for p in bench:
            print(f"  {p['position']:<5} {p['player_display_name']:<24} {p['mean']:5.1f}")

    # ---- Warnings -----------------------------------------------------------------------------
    warnings = list(notes)
    for r in team.to_dicts():
        if r["mean"] is None:
            warnings.append(f"{r['player_display_name']}: no projection (bye, inactive, or too few games)")
        inj = injury_line(r["player_id"], season, week)
        if any(s in inj for s in ("Out", "Doubtful", "Questionable")):
            warnings.append(f"{r['player_display_name']}: {inj}")
    for b in bench:
        for slot, s in lineup:
            # only compare players who could actually swap into the same slot
            if s and b["position"] in FLEX_OK.get(slot, {slot}) \
                    and abs(s["mean"] - b["mean"]) <= CLOSE_CALL:
                safer = s if s["floor"] >= b["floor"] else b
                boomier = b if safer is s else s
                warnings.append(
                    f"Close call: {s['player_display_name']} ({s['mean']:.1f}) vs "
                    f"{b['player_display_name']} ({b['mean']:.1f}). Safer: {safer['player_display_name']}, "
                    f"more upside: {boomier['player_display_name']}"
                )
                break
    if warnings:
        print("\nNOTES")
        for w in warnings:
            print(f"  - {w}")
    print()


if __name__ == "__main__":
    main()