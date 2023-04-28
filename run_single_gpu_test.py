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

for i in range(4):
    print('================')
    for j in range(8):
        tmpPara = copy.deepcopy(paraArray)
        tmpPara [paraArray[-2]] = paraArray[paraArray[-2]] * 2**i
        tmpPara [paraArray[-1]] = paraArray[paraArray[-1]] * 2**j
        cmdPara = " ".join(str(item) for item in tmpPara)[:-3]
        cmdPara = str(j) + ' ' + cmdPara + str(j+60000) + ' ' +hosts[i]+' '
        #print(cmdPara)
        #cmd = f"docker exec dps_node  bash -c 'cd /mnt/md0/db/workspace/cw_train_3 && ./8tp.sh {cmdPara} ' &"
        cmd = f"./8tp.sh {cmdPara} &"
        print(cmd)

        os.system(f"""ssh root@{hosts[i]} "docker exec dps_node  bash -c 'cd /mnt/md0/db/workspace/cw_train_3 && ./8tp.sh {cmdPara} ' " &""")
    os.system("sleep 30")
