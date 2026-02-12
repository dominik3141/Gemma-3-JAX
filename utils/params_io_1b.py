import jax
import orbax.checkpoint as ocp
import os
import logging


LOGGER = logging.getLogger(__name__)

Params = dict[str, jax.Array]

# get parameter only from gcp bucket if not running locally
if os.environ.get("GEMMA_RUNNING_LOCALLY") == "true":
    LOGGER.info("Running locally, using local checkpoint")
    DEFAULT_ORBAX_CHECKPOINT = os.path.abspath("data/gemma-3-1b/gemma-3-1b-pt-orbax")
else:
    LOGGER.info("Running on GCP, using GCP checkpoint")
    DEFAULT_ORBAX_CHECKPOINT = "gs://gemma-3-1b-pt-orbax-b76114af/gemma-3-1b-pt-orbax"


def load_params(path: str) -> Params:
    "Use orbax to load parameters"
    ckpt = ocp.StandardCheckpointer().restore(path)

    return ckpt
