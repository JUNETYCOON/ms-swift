import json
import urllib.request

from alibabacloud_pai_dlc20201203 import models as dlc_models
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
    request = dlc_models.ListJobsRequest(
        workspace_id="280441",
        resource_id="quota1pymm163y7p",
        page_number=1,
        page_size=30,
    )
    response = client.list_jobs(request)
    payload = response.to_map()
    body = payload.get("body")
    body_map = body.to_map() if hasattr(body, "to_map") else body
    if not isinstance(body_map, dict):
        print("BODY_TYPE", type(body), body_map)
        return 1
    print("RESPONSE_KEYS", sorted(body_map.keys()))
    jobs = body_map.get("Jobs") or body_map.get("jobs") or []
    for job in jobs:
        print(
            json.dumps(
                {
                    "JobId": job.get("JobId"),
                    "DisplayName": job.get("DisplayName"),
                    "Status": job.get("Status"),
                    "StartedAt": job.get("StartedAt"),
                    "FinishedAt": job.get("FinishedAt"),
                },
                ensure_ascii=False,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
