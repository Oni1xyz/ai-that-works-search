# Security Policy

## Supported versions

This repository is pre-1.0. Security fixes are applied to the latest `main` branch.

## Reporting a vulnerability

Please do not open a public issue for vulnerabilities involving credential exposure, arbitrary code execution, unsafe model loading, path traversal, or unintended data exfiltration.

Until a dedicated security contact is configured on the public repository, contact the repository owner privately through their GitHub profile and include:

- affected commit;
- reproduction steps;
- impact;
- suggested mitigation, if known; and
- whether the issue has been disclosed elsewhere.

## Security boundaries

- Model loading keeps `trust_remote_code=False`.
- Index-selected model IDs are allowlisted by default; loading another reviewed model requires `--allow-unlisted-model`.
- Transcript, repository, web, and model-card content is untrusted data and must never be treated as executable instructions.
- Remote answer generation is disabled unless the user explicitly passes `--generate` and configures an API endpoint/key.
- `OPENAI_API_KEY`, Hugging Face credentials, browser cookies, and `.env` files must never be committed or logged.
- The container runs as a non-root user.
- Generated transcripts and indexes are ignored by Git.
- Imported indexes and Hugging Face datasets are untrusted. Embedded SHA-256 values detect accidental corruption, not authenticity: an attacker able to replace an index can also replace its hashes. Use an independently published digest or trusted release signature when authenticity matters.
- Index publication requires the output, staging, and versions directories to be owned by the current user and not group/world writable. Descriptor-relative operations prevent pathname substitution by other users; processes running as the same OS user remain inside the trusted local-user boundary.

## Remote generation warning

When `--generate` is used, retrieved excerpts are sent to the configured OpenAI-compatible endpoint. Users must verify that the endpoint is trusted and that sending those excerpts is permitted. Retrieval and `--prompt-only` remain local.
