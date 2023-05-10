import torch
import torch.nn.functional as F
import sys
import os


import datetime
from pytz import timezone

import pandas as pd

m = int(sys.argv[1])
k = int(sys.argv[2])
n = int(sys.argv[3])
b = int(sys.argv[4])

csv_name = sys.argv[5]

active = 10

zone = 'Asia/Shanghai'
now_time = datetime.datetime.now(timezone(zone))
print (now_time.strftime("%Y-%m-%d %H:%M:%S"))
print(f'hello torch cuda: M:{m},N:{n},K:{k}')
cuda = torch.device('cuda')

tflops = 0
#us
self_cuda_time = 0

# estimate how much time to do the work, at least 10S

times = int(40*312*(10**12)/(b*m*n*k*2))

with torch.cuda.device(1):
    x = torch.rand((b, m , k), device=cuda, dtype=torch.bfloat16)
    y = torch.rand((b, k, n), device=cuda, dtype=torch.bfloat16)
    #z = torch.rand((m , n), device=cuda, dtype=torch.bfloat16)
    z = torch.rand((b, m, n), device=cuda, dtype=torch.bfloat16)
    # Non-default profiler schedule allows user to turn profiler on and off
    # on different iterations of the training loop;
    # trace_handler is called every time a new trace becomes available
    def trace_handler(prof):
        print(prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1))
        for item in prof.key_averages():
            if item.key == 'aten::baddbmm':
                flops = item.flops
                global tflops
                global self_cuda_time
                tflops = flops /1000000000000
                self_cuda_time = item.self_cuda_time_total /active



        # prof.export_chrome_trace("/tmp/test_trace_" + str(prof.step_num) + ".json")

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],

        # In this example with wait=1, warmup=1, active=2, repeat=1,
        # profiler will skip the first step/iteration,
        # start warming up on the second, record
        # the third and the forth iterations,
        # after which the trace will become available
        # and on_trace_ready (when set) is called;
        # the cycle repeats starting with the next step

        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=active,
            repeat=1),
        on_trace_ready=trace_handler,
        with_flops=True
        # on_trace_ready=torch.profiler.tensorboard_trace_handler('./log')
        # used when outputting for tensorboard
        ) as p:
            for step in range(max(times,20)):

                z = torch.baddbmm(z, x, y)
                # send a signal to the profiler that the next iteration has started
                p.step()



    print(z)
now_time = datetime.datetime.now(timezone(zone))
print (now_time.strftime("%Y-%m-%d %H:%M:%S"))
d = {'B': [b], 'M': [m], 'K':[k], 'N':[n], 'Flops':[tflops], 'SOL':[tflops/active*1000000/self_cuda_time/312], 'TimeStamp':[now_time.strftime("%Y-%m-%d %H:%M:%S")]}
df = pd.DataFrame(data=d)
output_path=f'{csv_name}_result.csv'
print(output_path)
df.to_csv(output_path, mode='a', header=not os.path.exists(output_path))
