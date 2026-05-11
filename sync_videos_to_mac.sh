#!/usr/bin/env bash
# Run on your MAC. Pulls jamebot training videos from bedroom-desktop into
# ~/jamebot_videos/<run_dir>/, keeping each experiment in its own folder.
#
# Install on the Mac:
#   scp bedroom-desktop:~/git/jamebot/sync_videos_to_mac.sh ~/sync_jamebot_videos.sh
#   chmod +x ~/sync_jamebot_videos.sh
#
# Usage on the Mac:
#   ~/sync_jamebot_videos.sh                # pull all runs
#   ~/sync_jamebot_videos.sh 2026-05-10*    # only matching runs (glob is remote-side)
#
# Env overrides:
#   REMOTE          SSH alias (default: bedroom-desktop)
#   REMOTE_LOG_DIR  remote IsaacLab log dir (default: git/IsaacLab/logs/rsl_rl/jamebot_balance)
#   LOCAL_DIR       local destination (default: ~/jamebot_videos)
set -euo pipefail

REMOTE="${REMOTE:-bedroom-desktop}"
REMOTE_LOG_DIR="${REMOTE_LOG_DIR:-git/IsaacLab/logs/rsl_rl/jamebot_balance}"
LOCAL_DIR="${LOCAL_DIR:-$HOME/jamebot_videos}"
PATTERN="${1:-*}"

mkdir -p "$LOCAL_DIR"

# Enumerate matching run dirs on the remote (bash 3.2-compatible).
runs=()
while IFS= read -r line; do
    [[ -n "$line" ]] && runs+=("$line")
done < <(
    ssh "$REMOTE" "cd '$REMOTE_LOG_DIR' 2>/dev/null && ls -1d $PATTERN/ 2>/dev/null" \
        | sed 's:/$::'
)

if [[ ${#runs[@]} -eq 0 ]]; then
    echo "No runs matching '$PATTERN' under $REMOTE:$REMOTE_LOG_DIR"
    exit 0
fi

echo "Found ${#runs[@]} run(s) on $REMOTE."
for run in "${runs[@]}"; do
    src="$REMOTE:$REMOTE_LOG_DIR/$run/videos/train/"
    dst="$LOCAL_DIR/$run/"
    echo "==> $run"
    mkdir -p "$dst"
    # --progress (not --info=progress2) for compatibility with macOS's
    # stock rsync 2.6.9. Errors aren't suppressed — if a run has no videos
    # yet you'll see rsync's "No such file or directory" and the loop
    # continues.
    rsync -avh --partial --progress \
        --include="*.mp4" --exclude="*" \
        "$src" "$dst" || echo "    (skipped: rsync exit $?)"
done

echo
echo "Done. Videos in $LOCAL_DIR"
