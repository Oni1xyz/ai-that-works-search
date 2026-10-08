# Security Policy

## Supported versions

This repository is pre-1.0. Security fixes are applied to the latest `main` branch.

## Reporting a vulnerability

Do not report the following vulnerabilities in a public issue: credential exposure, arbitrary code execution, unsafe model loading, path traversal, or unintended data exfiltration.

Until the repository has a dedicated security contact, report the vulnerability privately to the repository owner. Use the owner's GitHub profile. Include the following information:

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
- Treat imported indexes and Hugging Face datasets as untrusted. Embedded SHA-256 values detect accidental corruption. The values do not prove authenticity. An attacker who replaces an index can also replace its hashes. If authenticity matters, verify an independently published digest or a trusted release signature.
- Before publication, verify that the current user owns the output, staging, and versions directories. Verify that these directories are not group-writable or world-writable. Descriptor-relative operations prevent other users from substituting pathnames. Processes that run as the same operating-system user remain inside the trusted local-user boundary.

## Remote generation warning

Before you use `--generate`, verify that you trust the configured OpenAI-compatible endpoint. Confirm that you may send the retrieved excerpts to that endpoint. The command sends the excerpts to the endpoint. Retrieval and `--prompt-only` remain local.
