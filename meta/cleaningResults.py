import os
os.chdir("..")

resultsDir = 'Results/origComp3'

# paths = ['Ackley/2dim', 'Ackley/5dim.', 'Ackley/10dim', 'Ackley/20dim']
# paths = ['Ackley/2dim', 'Ackley/5dim', 'Ackley/10dim', 'Ackley/20dim',
#          'Griewank/2dim', 'Griewank/5dim', 'Griewank/10dim', 'Griewank/20dim',
#          'Rastrigin/2dim', 'Rastrigin/5dim', 'Rastrigin/10dim', 'Rastrigin/20dim']
# paths = ['AckleyRandom/2dim', 'AckleyRandom/5dim', 'AckleyRandom/10dim', 'AckleyRandom/20dim',
#          'GriewankRandom/2dim', 'GriewankRandom/5dim', 'GriewankRandom/10dim', 'GriewankRandom/20dim',
#          'RastriginRandom/2dim', 'RastriginRandom/5dim', 'RastriginRandom/10dim', 'RastriginRandom/20dim']

paths = ['Ackley/2dim', 'Ackley/5dim', 'Ackley/10dim', 'Ackley/20dim',
         'Griewank/2dim', 'Griewank/5dim', 'Griewank/10dim', 'Griewank/20dim',
         'Rastrigin/2dim', 'Rastrigin/5dim', 'Rastrigin/10dim', 'Rastrigin/20dim',
         'AckleyRandom/2dim', 'AckleyRandom/5dim', 'AckleyRandom/10dim', 'AckleyRandom/20dim',
         'GriewankRandom/2dim', 'GriewankRandom/5dim', 'GriewankRandom/10dim', 'GriewankRandom/20dim',
         'RastriginRandom/2dim', 'RastriginRandom/5dim', 'RastriginRandom/10dim', 'RastriginRandom/20dim']

for path in paths:
    pathName = resultsDir + "/" + path
    for fileName in os.listdir(pathName):
        filePos = pathName + "/" + fileName
        # for deleting everything that's not a param file
        # if fileName != "params.txt":
        #     os.unlink(filePos)

        # for setting all allocSizes to 1
        # if fileName == "params.txt":
        #     with open(filePos, 'r') as g:
        #         data = g.readlines()
        #     for index, line in enumerate(data):
        #         if line.startswith('allocSize'):
        #             data[index] = 'allocSize = 1\n'
        #     with open(filePos, 'w') as file:
        #         file.writelines(data)

