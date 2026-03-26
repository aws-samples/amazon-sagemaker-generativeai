import os
import re
from dataclasses import dataclass, field
from typing import List, Optional

import boto3
import yaml

SHARE_ID_PATTERN = re.compile(r"^[a-zA-Z0-9]+\*?$")

SERVICE_JOB_TYPE = "SAGEMAKER_TRAINING"

VALID_QUEUE_TYPES = {"fifo", "fair-share", "quota-management"}
VALID_SHARING_STRATEGIES = {"RESERVE", "LEND", "LEND_AND_BORROW"}

batch_client = boto3.client("batch")


@dataclass
class Resource:
    name: str
    arn: str


@dataclass
class SchedulingPolicyConfig:
    """Scheduling policy for fair-share queues."""

    name: str
    share_decay_seconds: int
    share_distribution: list

    def to_create_request(self):
        return {
            "name": self.name,
            "fairsharePolicy": {
                "shareDecaySeconds": self.share_decay_seconds,
                "shareDistribution": [
                    {
                        "shareIdentifier": s["identifier"],
                        "weightFactor": s["weight"],
                    }
                    for s in self.share_distribution
                ],
            },
        }


@dataclass
class QuotaManagementPolicyConfig:
    """Scheduling policy for quota-management queues."""

    name: str

    def to_create_request(self):
        return {
            "name": self.name,
            "quotaSharePolicy": {
                "idleResourceAssignmentStrategy": "FIFO",
            },
        }


@dataclass
class QuotaShareConfig:
    name: str
    capacity_limits: list  # [{"capacity_unit": "ml.g5.xlarge", "max_capacity": 1}]
    sharing_strategy: str  # RESERVE, LEND, LEND_AND_BORROW
    in_share_preemption: bool = False
    borrow_limit: Optional[int] = None  # percentage, only for LEND_AND_BORROW

    def to_create_request(self, job_queue_name):
        request = {
            "quotaShareName": self.name,
            "jobQueue": job_queue_name,
            "capacityLimits": [
                {
                    "maxCapacity": cl["max_capacity"],
                    "capacityUnit": cl["capacity_unit"],
                }
                for cl in self.capacity_limits
            ],
            "resourceSharingConfiguration": {
                "strategy": self.sharing_strategy,
            },
            "preemptionConfiguration": {
                "inSharePreemption": (
                    "ENABLED" if self.in_share_preemption else "DISABLED"
                ),
            },
            "state": "ENABLED",
        }
        if self.sharing_strategy == "LEND_AND_BORROW" and self.borrow_limit is not None:
            request["resourceSharingConfiguration"]["borrowLimit"] = self.borrow_limit
        return request


@dataclass
class ServiceEnvironmentConfig:
    name: str
    max_capacity: Optional[int] = None  # for fifo/fair-share (NUM_INSTANCES)
    capacity_limits: Optional[list] = None  # for quota-management (per instance type)

    def to_create_request(self):
        request = {
            "serviceEnvironmentName": self.name,
            "serviceEnvironmentType": SERVICE_JOB_TYPE,
            "state": "ENABLED",
        }
        if self.capacity_limits:
            request["capacityLimits"] = [
                {
                    "maxCapacity": cl["max_capacity"],
                    "capacityUnit": cl["capacity_unit"],
                }
                for cl in self.capacity_limits
            ]
        else:
            request["capacityLimits"] = [
                {"maxCapacity": self.max_capacity, "capacityUnit": "NUM_INSTANCES"}
            ]
        return request


@dataclass
class QueueConfig:
    name: str
    type: str  # "fifo", "fair-share", or "quota-management"
    priority: int
    service_environment: ServiceEnvironmentConfig
    scheduling_policy: Optional[SchedulingPolicyConfig] = None
    qm_policy: Optional[QuotaManagementPolicyConfig] = None
    quota_shares: Optional[List[QuotaShareConfig]] = None

    def to_create_request(self, scheduling_policy_arn=None):
        request = {
            "jobQueueName": self.name,
            "jobQueueType": SERVICE_JOB_TYPE,
            "state": "ENABLED",
            "priority": self.priority,
            "serviceEnvironmentOrder": [
                {"order": 1, "serviceEnvironment": self.service_environment.name},
            ],
        }
        if scheduling_policy_arn:
            request["schedulingPolicyArn"] = scheduling_policy_arn
        return request


@dataclass
class Resources:
    service_environments: List[Resource] = field(default_factory=list)
    scheduling_policies: List[Resource] = field(default_factory=list)
    job_queues: List[Resource] = field(default_factory=list)
    quota_shares: List[Resource] = field(default_factory=list)


def load_config(config_path=None):
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    queues = []
    for q in raw["queues"]:
        queue_type = q["type"]
        if queue_type not in VALID_QUEUE_TYPES:
            raise ValueError(
                f"Invalid queue type '{queue_type}' for queue '{q['name']}'. "
                f"Must be one of: {', '.join(sorted(VALID_QUEUE_TYPES))}"
            )

        # Parse service environment
        se_raw = q["service_environment"]
        if queue_type == "quota-management":
            if "capacity_limits" not in se_raw:
                raise ValueError(
                    f"Queue '{q['name']}' (quota-management) requires "
                    f"'capacity_limits' in service_environment."
                )
            se = ServiceEnvironmentConfig(
                name=se_raw["name"],
                capacity_limits=se_raw["capacity_limits"],
            )
        else:
            se = ServiceEnvironmentConfig(
                name=se_raw["name"],
                max_capacity=se_raw["max_capacity"],
            )

        # Parse scheduling policy
        sp = None
        qm_policy = None
        if queue_type == "fair-share":
            if not q.get("scheduling_policy"):
                raise ValueError(
                    f"Queue '{q['name']}' (fair-share) requires a 'scheduling_policy'."
                )
            sp_raw = q["scheduling_policy"]
            for s in sp_raw["share_distribution"]:
                sid = s["identifier"]
                if not SHARE_ID_PATTERN.match(sid):
                    raise ValueError(
                        f"Invalid share identifier '{sid}' in queue '{q['name']}'. "
                        f"Only alphanumeric characters (and an optional trailing *) are allowed."
                    )
            sp = SchedulingPolicyConfig(
                name=sp_raw["name"],
                share_decay_seconds=sp_raw.get("share_decay_seconds", 0),
                share_distribution=sp_raw["share_distribution"],
            )
        elif queue_type == "quota-management":
            if not q.get("scheduling_policy"):
                raise ValueError(
                    f"Queue '{q['name']}' (quota-management) requires a 'scheduling_policy'."
                )
            qm_policy = QuotaManagementPolicyConfig(
                name=q["scheduling_policy"]["name"],
            )

        # Parse quota shares
        quota_shares = None
        if queue_type == "quota-management":
            if not q.get("quota_shares"):
                raise ValueError(
                    f"Queue '{q['name']}' (quota-management) requires at least one 'quota_shares' entry."
                )
            quota_shares = []
            for qs_raw in q["quota_shares"]:
                strategy = qs_raw["sharing_strategy"]
                if strategy not in VALID_SHARING_STRATEGIES:
                    raise ValueError(
                        f"Invalid sharing_strategy '{strategy}' for quota share '{qs_raw['name']}'. "
                        f"Must be one of: {', '.join(sorted(VALID_SHARING_STRATEGIES))}"
                    )
                if strategy == "LEND_AND_BORROW" and "borrow_limit" not in qs_raw:
                    raise ValueError(
                        f"Quota share '{qs_raw['name']}' with LEND_AND_BORROW strategy "
                        f"requires a 'borrow_limit'."
                    )
                if not qs_raw.get("capacity_limits"):
                    raise ValueError(
                        f"Quota share '{qs_raw['name']}' requires at least one 'capacity_limits' entry."
                    )
                if len(qs_raw["capacity_limits"]) > 5:
                    raise ValueError(
                        f"Quota share '{qs_raw['name']}' can have at most 5 capacity limits."
                    )
                quota_shares.append(
                    QuotaShareConfig(
                        name=qs_raw["name"],
                        capacity_limits=qs_raw["capacity_limits"],
                        sharing_strategy=strategy,
                        in_share_preemption=qs_raw.get("in_share_preemption", False),
                        borrow_limit=qs_raw.get("borrow_limit"),
                    )
                )

        queues.append(
            QueueConfig(
                name=q["name"],
                type=queue_type,
                priority=q["priority"],
                service_environment=se,
                scheduling_policy=sp,
                qm_policy=qm_policy,
                quota_shares=quota_shares,
            )
        )

    return queues
