---
name: demo
description: A demonstration skill. Load it to see how progressive-disclosure skills work in this harness.
---

# Demo skill

This is the body of the demo skill. It only loads into context when the model
calls `load_skill("demo")`.

When this skill is active:
- Greet the user with "🛠 demo skill active".
- Explain that skills are folders under `./skills/` containing a `SKILL.md` with
  YAML frontmatter (`name`, `description`) and a markdown body.

Add bundled scripts or reference files alongside this `SKILL.md` and mention them
here so the model knows to read or run them via the `read_file` / `bash` tools.
