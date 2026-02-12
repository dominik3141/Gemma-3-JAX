# AGENTS

## Commit policy (do not skip)
- Agents must never create commits unless the user explicitly instructs them to commit.
- Keep all modifications in the working tree (staged or unstaged) until direct approval is given.

## TPU v5e runtime reminder (do not skip)
- For v5e (v5litepod) TPU VMs, use runtime version: `v2-alpha-tpuv5-lite`.
- Using `tpu-ubuntu2204-base` (or other v4/older runtimes) on v5e causes TPU init instability and watchdog timeouts.
- We have hit this multiple times. Always double check the runtime before creating a v5e TPU VM.

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

## Default region (do not skip)
- Default GCP region for new resources is `europe-west4`.
- Default TPU zone is `europe-west4-b` unless explicitly overridden.
