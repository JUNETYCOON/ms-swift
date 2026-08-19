"""Create a single-node 16-GPU bs8 VideoTrack v10 smoke with 800Gi memory."""

from __future__ import annotations

import json
import urllib.request

from alibabacloud_pai_dlc20201203 import models as dlc_models
from alibabacloud_pai_dlc20201203.client import Client
from alibabacloud_tea_openapi import models as open_api_models


CREDENTIAL_URL = "http://localhost:7002/api/v1/credentials/0"
REGION = "cn-wulanchabu"
ENDPOINT = f"pai-dlc.{REGION}.aliyuncs.com"
WORKSPACE_ID = "280441"
RESOURCE_ID = "quota1pymm163y7p"
IMAGE = "pudu-pai-registry-vpc.cn-wulanchabu.cr.aliyuncs.com/huojunliang/stage1-pengtaijun:stage1-723"
OSS_URI = "oss://pudu-pai.oss-cn-wulanchabu-internal.aliyuncs.com/luojunkun/"
MOUNT_PATH = "/mnt/luojunkun/"
SCRIPT = "stage1_mem_smoke_videotrack_bs8_sample300_v10.sh"
SCRIPT_DIR = "/mnt/luojunkun/stage1/ms-swift/scripts/smoke-videotrack-20260818/scripts"


def fetch_credentials() -> dict:
    with urllib.request.urlopen(CREDENTIAL_URL, timeout=5) as response:
        return json.load(response)


def make_client(credentials: dict) -> Client:
    config = open_api_models.Config(
        access_key_id=credentials["AccessKeyId"],
        access_key_secret=credentials["AccessKeySecret"],
        security_token=credentials["SecurityToken"],
        region_id=REGION,
        endpoint=ENDPOINT,
    )
    return Client(config)


def main() -> int:
    settings = dlc_models.JobSettings(
        allocate_all_rdmadevices=False,
        allow_unschedulable_nodes=False,
        disable_ecs_stock_check=False,
        enable_dswdev=False,
        enable_error_monitoring_in_aimaster=True,
        enable_oss_append=False,
        enable_rdma=True,
        enable_sanity_check=False,
        enable_tide_resource=False,
        error_monitoring_args=(
            "--job-execution-mode=Sync --enable-job-restart=True "
            "--max-num-of-job-restart=10 --fault-tolerant-policy=OnFailure"
        ),
        oversold_type="ForbiddenQuotaOverSold",
        tags={
            "Purpose": "stage1-mem-smoke-videotrack-bs8-sample300-v10",
            "DisplayName": "videotrack-bs8-sample300-v10",
            "Dataset": "norm1000-sample300",
        },
    )
    job_spec = dlc_models.JobSpec(
        ecs_spec="",
        image=IMAGE,
        pod_count=1,
        resource_config=dlc_models.ResourceConfig(
            cpu="30",
            gpu="16",
            memory="800Gi",
            shared_memory="800Gi",
        ),
        type="Worker",
    )
    data_source = dlc_models.CreateJobRequestDataSources(
        data_source_id="",
        mount_path=MOUNT_PATH,
        uri=OSS_URI,
    )
    request = dlc_models.CreateJobRequest(
        accessibility="PRIVATE",
        data_sources=[data_source],
        description=(
            "Molmo2-VideoTrack bs8 v10 smoke: all 16 GPUs, no checkpoints, "
            "batch_trace per-step sample/loss/model-output logging and per-step point accuracy."
        ),
        display_name="mem-smoke-videotrack-bs8-sample300-v10-20260817",
        job_max_running_time_minutes=1440,
        job_specs=[job_spec],
        job_type="PyTorchJob",
        resource_id=RESOURCE_ID,
        settings=settings,
        user_command=f"cd /mnt/luojunkun/stage1/ms-swift\nbash {SCRIPT_DIR}/{SCRIPT}",
        workspace_id=WORKSPACE_ID,
    )
    client = make_client(fetch_credentials())
    response = client.create_job(request)
    print(json.dumps(response.to_map(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
