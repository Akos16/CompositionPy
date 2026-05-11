from dataHandler import dataHandler

test = dataHandler()
asd = test.getData()
#print(asd)
asd1 = test.getDataStats(asd)
for i in asd1[0]:
  print(i)
#for z in asd1[1]:
#    print(z)
#for j in asd1[2]:
 #   print(j)