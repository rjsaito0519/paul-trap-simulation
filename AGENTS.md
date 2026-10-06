# Repository guide for AI assistants

This repository plans a Paul trap simulator using C++17 for numerical kernels and Python for orchestration and visualization. CERN ROOT remains a storage option, not yet a settled requirement.

## Start here

- Read `README.md` for commands and the current directory layout.
- Read `docs/simulation/README.md` and the linked model, stability, architecture, validation, and roadmap documents before implementation.
- Inspect the files related to the request before editing them; do not scan or rewrite unrelated areas.

## Working rules

- Make the smallest change that fully solves the requested problem.
- Follow nearby code style and preserve compatibility unless a deliberate migration is requested.
- Do not introduce dependencies, directories, numerical kernels, or abstractions before the corresponding interface and validation criteria are documented.
- Do not hard-code user names, host names, absolute paths, dataset locations, or exact tool versions.
- Keep generated files, large data, ROOT outputs, and credentials out of Git.

## Physics and numerical safety

- Never change signs, units, coordinate conventions, dimensionless definitions, stability criteria, escape criteria, or initial-condition conventions silently.
- Keep Floquet stability distinct from finite-aperture, finite-time capture.
- State the expected numerical and physical effect of changes to integration, tolerances, grids, interpolation, or classification.
- Keep provenance needed to reproduce results, including Git revision, compiler and library versions, input configuration, random seed, solver, tolerances, grid, and observation time.

## Code and verification

- C++ uses C++17. Prefer RAII, values, standard-library facilities, and explicit ownership.
- Add C++ programs explicitly with `add_analysis_executable(...)`; do not glob sources into targets.
- Keep the validated equations and classification semantics in one authoritative implementation. Python reference calculations may remain separate only when used as an explicit cross-check.
- Build C++ changes with `./build.sh`. Run the smallest applicable benchmark and convergence test.
- Do not treat generated files under `.build*/`, `data/`, or `results/` as source files.

## Git commits

- Read-only commands such as `git status`, `git diff`, and `git log` may be used for inspection.
- Do not run `git add`, create a commit, or push unless the user explicitly requests that action.
- Present the proposed changes for review before staging or committing when practical.
- Use the repository user's configured Git identity as the sole author.
- Do not add AI identities or attribution such as `Co-authored-by`, `Made with`, or generated-by trailers.
- Do not change the user's Git name or email configuration.

## GitHub issues

- Proactively suggest an issue when a bug, deferred task, open question, or design decision should be tracked beyond the current work.
- Include a concise proposed title, context, acceptance criteria, and relevant files or evidence.
- A suggestion is not authorization: do not create, edit, label, assign, comment on, or close an issue until the user explicitly approves that action.
- After approval, keep the issue focused and avoid including credentials, private paths, or unnecessary environment-specific details.

## Documentation

- Keep `README.md` short and task-oriented.
- Put durable explanations in `docs/` and add them to `docs/README.md`.
- Keep the simulation documents under `docs/simulation/` and update their index when adding or replacing decisions.
- Give each topic directory its own `README.md` entry point and link it from `docs/README.md`.
- Record assumptions, units, input/output schemas, and validation methods for analysis logic.
- Date statements that describe temporary status or time-dependent results.
- Treat this file as the environment-independent source for AI guidance.
- Keep tool-specific settings such as `.cursor/` local and derive them from this file when needed.
