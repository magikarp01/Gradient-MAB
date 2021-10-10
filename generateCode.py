types = ["Fit", "Restless", "Trad"]
finities = ["", "Infinite"]
bais = ["OCBA", "UCB"]

index = 0
for i in types:
    for k in bais:
        for j in finities:
            print(f'if methods[{index}]:')
            print(f'    print(\"{i + j + k}\")')
            print(f'    multiprocessSearch(numProcesses, iterPerProcess, temp{i}{j}{k}, sharedParams, processStartPos, dir+"{i}{j}{k}.json")')
            print()
            index += 1

                  
                  
