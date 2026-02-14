# AGENTS

## Commit policy (do not skip)
- Agents must never create commits unless the user explicitly instructs them to commit.
- Keep all modifications in the working tree (staged or unstaged) until direct approval is given.

## Working style preferences (do not skip)
- Prefer simple defaults over optional configuration when there is one clear team path.
- Avoid adding flags/options "just in case" if they are unlikely to be used.
- For recurring ops workflows, hardcode practical project defaults instead of requiring repeated manual args.
- When the user asks to "set it up", execute the infra/app setup directly and verify it works end-to-end.

## Agent memory hygiene (do not skip)
- If you discover a durable, high-signal preference or workflow pattern that will help future agents succeed, add it to this file without waiting to be asked.
- Only add genuinely useful learnings; do not add one-off incidents, temporary context, or obvious noise.
- Keep additions concise and actionable.

## TPU v5e runtime reminder (do not skip)
- For v5e (v5litepod) TPU VMs, use runtime version: `v2-alpha-tpuv5-lite`.
- Using `tpu-ubuntu2204-base` (or other v4/older runtimes) on v5e causes TPU init instability and watchdog timeouts.
- We have hit this multiple times. Always double check the runtime before creating a v5e TPU VM.

## TPU v6e runtime reminder (do not skip)
- For v6e TPU VMs, use runtime version: `v2-alpha-tpuv6e`.

## Internal IP egress pitfall (do not skip)
- If TPU VMs are created with internal IPs only (`--internal-ips`), they do **not** have public internet egress by default.
- To allow outbound internet access (pip installs, model downloads, etc.), configure **Cloud NAT** in that region, backed by a **Cloud Router**, on the TPU VPC/subnet.
- `privateIpGoogleAccess: true` on the subnet is required for private access to Google APIs, but it is not a full internet egress replacement.
- Cloud Router is effectively free; Cloud NAT can incur charges (NAT gateway/data processing/external NAT IP), so create NAT where needed and clean up unused NATs.
- Example we already configured: router `trc-router-us-central1` + NAT `trc-nat-us-central1` in `us-central1` on network `default`.

## GCS bucket policy preference (do not skip)
- Prefer **uniform bucket-level access** for all buckets.
- Do not use fine-grained object ACLs unless explicitly requested for a specific bucket.
- When creating buckets via `gcloud storage buckets create`, include `--uniform-bucket-level-access`.

## Profiling logdir preference (do not skip)
- Prefer direct JAX profiler writes to GCS (`jax.profiler.start_trace("gs://...")`) instead of local trace staging + manual `gsutil cp`.
- Keep inference and RL traces under the same profiling logdir root so XProf/TensorBoard can discover sessions without custom stitching logic.
- When changing profiling layout conventions, clear old profile objects first to avoid mixed directory structures in the same bucket.

## Default region (do not skip)
- Default GCP region for new resources is `europe-west4`.
- Default TPU zone is `europe-west4-b` unless explicitly overridden.
