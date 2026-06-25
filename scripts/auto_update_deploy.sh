#!/usr/bin/env bash
set -Eeuo pipefail

export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"

REPO_DIR="${REPO_DIR:-/Users/wukong/mylife/grok2api}"
BRANCH="${BRANCH:-main}"
ORIGIN_REMOTE="${ORIGIN_REMOTE:-origin}"
UPSTREAM_REMOTE="${UPSTREAM_REMOTE:-upstream}"
UPSTREAM_URL="${UPSTREAM_URL:-https://github.com/jiujiu532/grok2api.git}"
HEALTH_URL="${HEALTH_URL:-http://127.0.0.1:8000/health}"
LOG_FILE="${LOG_FILE:-$REPO_DIR/logs/auto_update_deploy.log}"
LOCK_DIR="${LOCK_DIR:-$REPO_DIR/.auto_update_deploy.lock}"

mkdir -p "$(dirname "$LOG_FILE")"
exec >>"$LOG_FILE" 2>&1

log() {
  printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S%z')" "$*"
}

fail() {
  log "ERROR: $*"
  exit 1
}

if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  log "another auto update is already running; exiting"
  exit 0
fi
cleanup() {
  rmdir "$LOCK_DIR" 2>/dev/null || true
}
trap cleanup EXIT

cd "$REPO_DIR"

log "auto update check started"

if ! command -v git >/dev/null 2>&1; then
  fail "git is not available"
fi
if ! command -v docker >/dev/null 2>&1; then
  fail "docker is not available"
fi

current_branch="$(git branch --show-current)"
if [[ "$current_branch" != "$BRANCH" ]]; then
  fail "expected branch $BRANCH, current branch is $current_branch"
fi

if git remote get-url "$UPSTREAM_REMOTE" >/dev/null 2>&1; then
  git remote set-url "$UPSTREAM_REMOTE" "$UPSTREAM_URL"
else
  git remote add "$UPSTREAM_REMOTE" "$UPSTREAM_URL"
fi

git fetch "$ORIGIN_REMOTE" --prune
git fetch "$UPSTREAM_REMOTE" --prune

start_head="$(git rev-parse HEAD)"
local_head="$start_head"
upstream_head="$(git rev-parse "$UPSTREAM_REMOTE/$BRANCH")"
origin_head="$(git rev-parse "$ORIGIN_REMOTE/$BRANCH" 2>/dev/null || true)"

if ! git diff --quiet || ! git diff --cached --quiet || [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
  fail "working tree is dirty; commit local changes before unattended update"
fi

if [[ -n "$origin_head" && "$origin_head" != "$local_head" ]]; then
  if git merge-base --is-ancestor "$origin_head" HEAD; then
    log "local branch is ahead of $ORIGIN_REMOTE/$BRANCH"
  elif git merge-base --is-ancestor HEAD "$ORIGIN_REMOTE/$BRANCH"; then
    log "fast-forwarding local $BRANCH to $ORIGIN_REMOTE/$BRANCH before upstream sync"
    git merge --ff-only "$ORIGIN_REMOTE/$BRANCH"
    local_head="$(git rev-parse HEAD)"
    origin_head="$local_head"
  else
    fail "$ORIGIN_REMOTE/$BRANCH has commits not present locally; refusing to overwrite remote history"
  fi
fi

if git merge-base --is-ancestor "$UPSTREAM_REMOTE/$BRANCH" HEAD; then
  if [[ -n "$origin_head" && "$origin_head" != "$local_head" ]]; then
    log "pushing committed local changes to $ORIGIN_REMOTE/$BRANCH"
    git push "$ORIGIN_REMOTE" "HEAD:$BRANCH"
  else
    log "no upstream updates"
  fi
  exit 0
fi

backup_dir="$REPO_DIR/backups/auto/$(date '+%Y%m%d_%H%M%S')"
mkdir -p "$backup_dir"
cp data/config.toml "$backup_dir/config.toml"
cp data/accounts.db "$backup_dir/accounts.db"
log "backed up data to $backup_dir"

if git merge-base --is-ancestor HEAD "$UPSTREAM_REMOTE/$BRANCH"; then
  log "fast-forwarding $BRANCH to $UPSTREAM_REMOTE/$BRANCH"
  git merge --ff-only "$UPSTREAM_REMOTE/$BRANCH"
  log "pushing synced $BRANCH to $ORIGIN_REMOTE"
  git push "$ORIGIN_REMOTE" "HEAD:$BRANCH"
else
  log "rebasing committed local changes onto $UPSTREAM_REMOTE/$BRANCH"
  git rebase "$UPSTREAM_REMOTE/$BRANCH"
  new_head="$(git rev-parse HEAD)"
  log "pushing rebased $BRANCH to $ORIGIN_REMOTE with lease"
  git push --force-with-lease="$BRANCH:$origin_head" "$ORIGIN_REMOTE" "HEAD:$BRANCH"
  local_head="$new_head"
fi

log "building grok2api image"
docker compose build --pull grok2api

log "recreating grok2api service"
docker compose up -d --no-deps --force-recreate grok2api

log "waiting for health check"
for _ in $(seq 1 20); do
  if curl -fsS "$HEALTH_URL" >/dev/null; then
    log "deployment healthy: $HEALTH_URL"
    log "auto update completed: $start_head -> $(git rev-parse HEAD) (upstream $upstream_head)"
    exit 0
  fi
  sleep 3
done

fail "health check did not pass after deployment"
