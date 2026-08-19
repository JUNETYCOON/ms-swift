import json
import sys
import urllib.request

from alibabacloud_pai_dlc20201203 import models as dlc_models
from alibabacloud_pai_dlc20201203.client import Client
from alibabacloud_tea_openapi import models as open_api_models


CREDENTIAL_URL = "http://localhost:7002/api/v1/credentials/0"
REGION = "cn-wulanchabu"
ENDPOINT = f"pai-dlc.{REGION}.aliyuncs.com"


def main() -> int:
    job_ids = sys.argv[1:]
    with urllib.request.urlopen(CREDENTIAL_URL, timeout=5) as response:
        cred = json.load(response)
    config = open_api_models.Config(
        access_key_id=cred["AccessKeyId"],
        access_key_secret=cred["AccessKeySecret"],
        security_token=cred["SecurityToken"],
        region_id=REGION,
        endpoint=ENDPOINT,
    )
    client = Client(config)
    for job_id in job_ids:
        try:
            response = client.get_job(job_id, dlc_models.GetJobRequest())
            body = response.to_map().get("body")
            body_map = body.to_map() if hasattr(body, "to_map") else body
            print(json.dumps(body_map, ensure_ascii=False, indent=2, default=str))
        except Exception as error:  # noqa: BLE001
            print(f"{job_id} ERROR: {error}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
