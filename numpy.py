import numpy as np
l1=[1,2,3,4]
l2=[5,6,7,8]
na1=np.array(l1)
na2=np.array(l2)
na3=na1+na2
print(na1)
print(na2)
print(na3)

print(np.sum(na1))
print(np.sum(na2))
print(np.sum(na3))

print(np.mean(na1))
print(np.mean(na2))
print(np.mean(na3))

print(np.std(na1))
print(np.std(na2))
print(np.std(na3))