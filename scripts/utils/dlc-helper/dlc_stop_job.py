import json
import sys
import urllib.request

from alibabacloud_pai_dlc20201203.client import Client
from alibabacloud_tea_openapi import models as open_api_models


CREDENTIAL_URL = "http://localhost:7002/api/v1/credentials/0"
REGION = "cn-wulanchabu"
ENDPOINT = f"pai-dlc.{REGION}.aliyuncs.com"


def main() -> int:
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
    for job_id in sys.argv[1:]:
        response = client.stop_job(job_id)
        print(job_id, json.dumps(response.to_map(), ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
