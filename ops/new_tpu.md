## Step 0:
Cute resources requests for some reason stay around after they have been fulfilled. And since we always use the same name, we first have to check if there is still one existing and delete that one. 

{gcp command to delete node-1-request if it exists}


## Step 1:
Create request for TPU

gcloud compute tpus queued-resources create node-12-request \
  --node-id=node-12 \
  --zone=europe-west4-b \
  --accelerator-type=v5litepod-64 \
  --runtime-version=v2-alpha-tpuv5-lite \
  --spot \
  --internal-ips \
  --service-account=gemma-tpu-writer@default-482802.iam.gserviceaccount.com


## Step 2 
Wait for request to be fullfilled

## Step 3
Wait a minute for deployment to actually have been completed.
We have noticed before that if you try to connect to a fresh TPU too early, you will run into weird SSH errors. So you sometimes have to wait for a minute and sometimes have to wait for two. Therefore, we have to retry the following if we encounter any errors. 

uv run python -m ops.remote_sync --tpu --zone europe-west4-b --vm node-1

## Step 4
If Step 3 was succesfull and everything has been compied, we now do the setup

gcloud alpha compute tpus tpu-vm ssh node-1 --worker=all --command="cd app/ && python setup.py" --tunnel-through-iap

## Step 5
Training

gcloud alpha compute tpus tpu-vm ssh node-1 --worker=all --command="cd app/ && nohup ~/.local/bin/uv run python -m main" --tunnel-through-iap

## Step 6
Remind user to delete TPU when done

