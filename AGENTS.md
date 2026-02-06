# AGENTS

## TPU v5e runtime reminder (do not skip)
- For v5e (v5litepod) TPU VMs, use runtime version: `v2-alpha-tpuv5-lite`.
- Using `tpu-ubuntu2204-base` (or other v4/older runtimes) on v5e causes TPU init instability and watchdog timeouts.
- We have hit this multiple times. Always double check the runtime before creating a v5e TPU VM.
