# AGENTS

## Runtime

- We always use `uv` in this repository. Don't use the OS python unless you have a good reason to.

## Commit policy

- Agents must never create commits unless the user explicitly instructs them to commit.
- Keep all modifications in the working tree (staged or unstaged) until direct approval is given.

## Working style preferences

- Prefer simple defaults over optional configuration when there is one clear team path.
- Avoid adding flags/options "just in case" if they are unlikely to be used.
- For recurring ops workflows, hardcode practical project defaults instead of requiring repeated manual args.
- When the user asks to "set it up", execute the infra/app setup directly and verify it works end-to-end.
- Keep functions single-sequence when possible and apply batching at call sites via `jax.vmap`. In this repository, we use `vmap` whenever possible.
- Prefer `jaxtyping` annotations for JAX-heavy function signatures when practical.
- Do not use type aliases in function signatures; keep signatures explicit so they are readable at first sight.
- Prefer fail-fast behavior over defensive fallbacks when invalid inputs indicate a caller bug. Do not silently clamp, coerce, or return placeholder outputs just to avoid errors unless explicitly requested.

## Agent memory hygiene

- If you discover a durable, high-signal preference or workflow pattern that will help future agents succeed, add it to this file without waiting to be asked.
- Only add genuinely useful learnings; do not add one-off incidents, temporary context, or obvious noise.
- Keep additions concise and actionable.

## TPU v5e runtime reminder

- For v5e (v5litepod) TPU VMs, use runtime version: `v2-alpha-tpuv5-lite`.
- Using `tpu-ubuntu2204-base` (or other v4/older runtimes) on v5e causes TPU init instability and watchdog timeouts.
- We have hit this multiple times. Always double check the runtime before creating a v5e TPU VM.

## TPU v6e runtime reminder

- For v6e TPU VMs, use runtime version: `v2-alpha-tpuv6e`.

## Internal IP egress pitfall

- If TPU VMs are created with internal IPs only (`--internal-ips`), they do **not** have public internet egress by default.
- To allow outbound internet access (pip installs, model downloads, etc.), configure **Cloud NAT** in that region, backed by a **Cloud Router**, on the TPU VPC/subnet.
- `privateIpGoogleAccess: true` on the subnet is required for private access to Google APIs, but it is not a full internet egress replacement.
- Cloud Router is effectively free; Cloud NAT can incur charges (NAT gateway/data processing/external NAT IP), so create NAT where needed and clean up unused NATs.
- Example we already configured: router `trc-router-us-central1` + NAT `trc-nat-us-central1` in `us-central1` on network `default`.

## GCS bucket policy preference

- Prefer **uniform bucket-level access** for all buckets.
- Do not use fine-grained object ACLs unless explicitly requested for a specific bucket.
- When creating buckets via `gcloud storage buckets create`, include `--uniform-bucket-level-access`.

## Default region

- Default GCP region for new resources is `europe-west4`.
- Default TPU zone is `europe-west4-b` unless explicitly overridden.

## Multihost JAX profiling reminder

- For multihost TPU profiling, set one shared `jax.profiler.ProfileOptions().session_id` across hosts (for example by broadcasting a host-0 timestamp).
- If each host gets a different session id, traces are written to different `plugins/profile/<session_id>` folders and TensorBoard cannot stitch one combined profile.

## TensorBoard VM reminder

- For large TPU profiles, do not run TensorBoard/XProf on the local Mac. Use the GCP VM workflow in `ops/tensorboard_vm_runbook.md`.
- Preferred VM is `tb-profile-euw4` in `europe-west4-b`; start it when needed and stop it when idle to control cost.
