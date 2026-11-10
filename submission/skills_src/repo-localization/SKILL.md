---
name: repo-localization
description: Ranks the functions, methods and classes of the Python repository in /workspace against an issue text or keywords (BM25 over path, qualified name and body). Use at the start of a task to find which files and symbols to read and edit before opening anything.
---

# Repo localization

Run once at the start of a task, then read the top candidates with `read_file`.

Call:

    run_skill_script(skill_name="repo-localization", file_path="scripts/localize.py",
                     args={"query": "<paste the issue title and the key sentences>", "top": 10})

Output: the top files, the top units as `path:start-end  qualname  (kind)` with the signature line, and for the top 3 units what they call and who calls them (matched by simple name, so homonyms are possible).

Tips:
- Put identifiers from the issue (function, class, parameter, error message words) in the query. Long issue text is fine.
- Run it a second time with different keywords (a symptom word, a module name) if the first list looks off.
- The ranking finds likely places, not the fix. A unit is a good candidate when its name or path matches the issue; confirm by reading its lines.
- Test files are not indexed.
