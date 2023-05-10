import os

import datetime
from pytz import timezone
zone = 'Asia/Shanghai'
now_time = datetime.datetime.now(timezone(zone))


base = 32768
for i in range(9):
    m = base //(2**i)
    for j in range(9):
        n = base //(2**j)
        for l in range(9):
            k = base //(2**l)
            cmd = ' '.join(['python', 'perf_pytorch.py',str(m), str(n), str(k)])
            print('#########################################')
            print(cmd)
            os.system(' '.join(['python', 'perf_pytorch.py',str(m), str(n), str(k), now_time.strftime("%Y-%m-%d %H:%M:%S")]))
            print('||||||||||||||||||||||||||||||||||||||||')
            os.system("sleep 30s")