# TensorBoard Profile VM Runbook (For Agents)

This runbook documents the stable way to inspect TPU profile runs without loading TensorBoard locally.

## Canonical setup

- Project: `default-482802`
- Instance: `tb-profile-euw4`
- Zone: `europe-west4-b`
- Current machine type: `e2-highmem-8` (62 GiB RAM)
- TensorBoard service: `tensorboard-profile`
- Service file: `/etc/systemd/system/tensorboard-profile.service`
- Wrapper entrypoint: `/home/dffarr/tensorboard_xprof_wrapper.py`
- Default logdir: `gs://gemma-3-training-profiles-20260207-165411-1d9c5e-euw4`

## Start and stop the VM

Start:

```bash
gcloud compute instances start tb-profile-euw4 --zone=europe-west4-b
```

Stop (do this when done to avoid unnecessary cost):

```bash
gcloud compute instances stop tb-profile-euw4 --zone=europe-west4-b
```

Status:

```bash
gcloud compute instances describe tb-profile-euw4 --zone=europe-west4-b --format='value(status,machineType.basename(),networkInterfaces[0].accessConfigs[0].natIP)'
```

## Open TensorBoard from local browser

Create tunnel:

```bash
gcloud compute ssh tb-profile-euw4 --zone=europe-west4-b -- -N -L 16006:localhost:6006
```

Then open:

- `http://localhost:16006/#profile`

## Service health checks

From local machine:

```bash
gcloud compute ssh tb-profile-euw4 --zone=europe-west4-b --command 'systemctl is-active tensorboard-profile'
```

Check profile runs via tunnel:

```bash
curl --compressed -s http://127.0.0.1:16006/data/plugin/profile/runs
```

For a specific run:

```bash
RUN=gemma_inference_1668461314
curl --compressed -s "http://127.0.0.1:16006/data/plugin/profile/tools?run=${RUN}"
curl --compressed -s "http://127.0.0.1:16006/data/plugin/profile/hosts?run=${RUN}&tag=memory_profile"
```

## Known gotchas (important)

1. If a run is visible in the dropdown but tools show `No Data`, open `Overview Page` first for that run.
2. The wrapper initializes XProf worker stubs before TensorBoard starts. Without it, requests fail with `No worker service stub available`.
3. `overview_page` with `host=ALL_HOSTS` can OOM on large 16-host runs. The wrapper patches host selection so `overview_page` uses one host to stay stable.
4. Because of (3), overview is single-host. Use tool-specific host views (`Memory Profile`, `Op Profile`, etc.) for deeper analysis.

## Change TensorBoard logdir

If data is in a different bucket/path, update service `ExecStart` `--logdir`, then restart:

```bash
gcloud compute ssh tb-profile-euw4 --zone=europe-west4-b --command "\
sudo sed -i 's#--logdir [^ ]*#--logdir gs://YOUR_BUCKET_OR_PREFIX#' /etc/systemd/system/tensorboard-profile.service && \
sudo systemctl daemon-reload && \
sudo systemctl restart tensorboard-profile && \
systemctl is-active tensorboard-profile"
```

## Troubleshooting

Last logs:

```bash
gcloud compute ssh tb-profile-euw4 --zone=europe-west4-b --command 'sudo journalctl -u tensorboard-profile -n 200 --no-pager'
```

Look for:

- `No worker service stub available`: wrapper was not used or failed early.
- `Failed with result 'oom-kill'`: machine too small for requested conversion path.

## Memory planning guidance

Observed on this setup:

- TensorBoard child process RSS around `~12 GiB`
- Peak (`VmHWM`) around `~16 GiB`

Practical sizing:

1. With wrapper workaround: use at least `32 GiB` VM RAM.
2. Without workaround (`ALL_HOSTS` overview): can exceed `62 GiB` and still OOM.

## Stopped-state cost (keep VM, stop compute)

If you stop `tb-profile-euw4` without deleting it, CPU/RAM charges stop, but disk
storage remains billed.

Current configuration:

- Boot disk: `80 GiB` `pd-standard` (zonal) in `europe-west4`
- Regional price used: `$0.044 / GiB-month` (SKU `AE8C-46C3-4994`, effective `2026-02-15`)

Approximate cost:

1. Monthly disk cost: `80 * 0.044 = $3.52 / month`
2. Daily equivalent: about `$0.12 / day`

Notes:

- This estimate assumes no additional snapshots.
- If a reserved static external IP is attached while stopped, that can add extra cost.
