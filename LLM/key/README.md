# API keys (local-only)
- `gemini_key.txt` and `openai_key.txt` live here for local runs.
- The `.gitignore` excludes `LLM/key/*.txt`, so real keys never reach git or GitHub.
- If you need to set up keys on a new machine: `echo "your-key" > LLM/key/gemini_key.txt` (same for `openai_key.txt`).
- Prefer environment variables in scripts when possible (e.g., `export GEMINI_API_KEY=...`, `export OPENAI_API_KEY=...`), and only write files on trusted machines.
