#!/bin/zsh
project_dir=${0:A:h}
cd "$project_dir" || exit 1
if [[ ! -x .venv/bin/python ]]; then
  print 'The project Python environment is missing. Complete project setup first.'
  read -r 'task_response?Press Enter to close.'
  exit 1
fi
if .venv/bin/python -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8765/api/health",timeout=2)' >/dev/null 2>&1; then
  open 'http://127.0.0.1:8765'
  exit 0
fi
print 'Starting PipelineHub Chat. Keep this window open while chatting.'
(
  for attempt in {1..20}; do
    if .venv/bin/python -c 'import urllib.request; urllib.request.urlopen("http://127.0.0.1:8765/api/health",timeout=2)' >/dev/null 2>&1; then
      open 'http://127.0.0.1:8765'
      break
    fi
    sleep 1
  done
) &
.venv/bin/python -m cicd_support_rag.ui_server
