import os
import sys
import copy

def getIntPara(index):
    return int(sys.argv[index])
gpu_id = 0
global_batch = getIntPara(1)
micro_batch = getIntPara(2)
layers = getIntPara(3)
hidden = getIntPara(4)
seq = getIntPara(5)



paraArray = []
for i in range(len(sys.argv) - 1):
    paraArray.append(getIntPara(i+1))

print(paraArray)


hosts = {}
for i in range(4):
   hosts[i] = "llm00" + str(i+1)

print(hosts)

t = [1,2,4,8]
p = [8,4,2,1]

for i in range(4):
    print('================')
    tmpPara = copy.deepcopy(paraArray)
    tmpPara.append(t[i])
    tmpPara.append(p[i])
    cmdPara = " ".join(str(item) for item in tmpPara)
    #print(cmdPara)
    #cmd = f"docker exec dps_node  bash -c 'cd /mnt/md0/db/workspace/cw_train_3 && ./8tp.sh {cmdPara} ' &"
    cmd = f"./single_node.sh {cmdPara} &"
    print(cmd)

    os.system(f"""ssh root@{hosts[i]} "docker exec dps_node  bash -c 'cd /mnt/md0/db/workspace/cw_train_3 && ./single_node.sh {cmdPara} ' " &""")
    os.system("sleep 30")
