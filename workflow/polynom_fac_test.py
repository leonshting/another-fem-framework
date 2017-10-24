from common import polynom_factory
import numpy as np
ff = polynom_factory.local_funcfunc_matrix(order=2, dim=2, distribution='globatto')
cum = (ff.sum(axis=1))
lfg = polynom_factory.local_gradgrad_matrix(order=2, distribution='globatto', dim=2)

lgfg = polynom_factory.local_gradfunc_matrix(order=2, distribution='globatto', dim=2)[1][0]
print(lgfg)
#res = np.dot(np.linalg.inv(ff), lfg[0])
ress = [lfg[0], lgfg]
point = (0,0)
root = 0
for res in ress:
 x_der = 0
 xx_der = 0
 yy_der = 0
 y_der = 0
 xy_der = 0

 for k,v in lfg[1].items():

    dx = point[0] - k[0]
    dy = point[1] - k[1]

    x_der += dx*res[root,v]
    xx_der += res[root,v]*dx**2/2

    y_der += dy*res[root,v]
    yy_der += dy**2/2*res[root,v]

    xy_der += dx*dy*res[root,v]


 print(x_der/cum[root])
 print(y_der/cum[root])
 print(xx_der/cum[root])
 print(yy_der/cum[root])
 print(xy_der/cum[root])


