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

# Enumerate run dirs that *actually contain mp4s* under videos/train.
# One SSH call instead of one-per-run, and we skip the rsync entirely
# for runs that haven't recorded a video yet (saves a few seconds per
# empty run and silences the "No such file or directory" spam).
runs=()
while IFS= read -r line; do
    [[ -n "$line" ]] && runs+=("$line")
done < <(
    ssh "$REMOTE" "cd '$REMOTE_LOG_DIR' 2>/dev/null || exit 0; \
        for d in $PATTERN/; do \
            [ -d \"\$d/videos/train\" ] || continue; \
            set -- \"\$d/videos/train/\"*.mp4; \
            [ -e \"\$1\" ] && echo \"\${d%/}\"; \
        done"
)

if [[ ${#runs[@]} -eq 0 ]]; then
    echo "No runs matching '$PATTERN' with videos under $REMOTE:$REMOTE_LOG_DIR"
    exit 0
fi

echo "Found ${#runs[@]} run(s) with videos on $REMOTE."
for run in "${runs[@]}"; do
    src="$REMOTE:$REMOTE_LOG_DIR/$run/videos/train/"
    dst="$LOCAL_DIR/$run/"
    echo "==> $run"
    mkdir -p "$dst"
    # --progress (not --info=progress2) for macOS's stock rsync 2.6.9.
    rsync -avh --partial --progress \
        --include="*.mp4" --exclude="*" \
        "$src" "$dst" || echo "    (rsync exit $?)"
done

echo
echo "Done. Videos in $LOCAL_DIR"
