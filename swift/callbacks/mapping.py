# Copyright (c) ModelScope Contributors. All rights reserved.
from .activation_cpu_offload import ActivationCpuOffloadCallBack
from .adalora import AdaloraCallback
from .batch_trace import BatchTraceCallback
from .deepspeed_elastic import DeepspeedElasticCallback, GracefulExitCallback
from .early_stop import EarlyStopCallback
from .lisa import LISACallback
from .perf_log import PerfMetricsLogCallback

callbacks_map = {
    'activation_cpu_offload': ActivationCpuOffloadCallBack,
    'adalora': AdaloraCallback,
    'batch_trace': BatchTraceCallback,
    'deepspeed_elastic': DeepspeedElasticCallback,
    'early_stop': EarlyStopCallback,
    'graceful_exit': GracefulExitCallback,
    'lisa': LISACallback,
    'perf_log': PerfMetricsLogCallback
}
