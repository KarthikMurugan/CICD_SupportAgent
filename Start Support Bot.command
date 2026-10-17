#!/bin/zsh
# Open this file from Finder to ask support questions in Terminal.
project_dir=${0:A:h}
cd "$project_dir" || exit 1

if [[ ! -x .venv/bin/python ]]; then
  print 'The project Python environment is missing. Complete project setup first.'
  read -r 'task_response?Press Enter to close.'
  exit 1
fi

print 'PipelineHub Support Assistant'
print 'Answers use Pinecone evidence and the local Ollama model.'
print 'Keep Ollama running. Submit an empty question to quit.'

while true; do
  print ''
  IFS= read -r 'support_question?Your question: ' || break
  [[ -z "$support_question" ]] && break
  .venv/bin/python -m cicd_support_rag.bot "$support_question"
done
