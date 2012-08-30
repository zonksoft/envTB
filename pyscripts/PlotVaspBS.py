#!/usr/bin/env python
import sys,csv
#http://gnuplot-py.sourceforge.net/
sys.path.append('/home/lv70071/reiter/pythontest/gnuplot-py-1.8/build/lib')
import Gnuplot
#deprecated!
file = sys.argv[1]
data = [line.split() for line in open(file,'r').read().splitlines()]

nrelectrons,nrkpoints,nrbands=[int(x) for x in data[5]]

allkpointcoords=[]
allenergies=[]
for kpointfirstline in [7+kpoint*(nrbands+2) for kpoint in range(nrkpoints)]:
	allkpointcoords.append(data[kpointfirstline])
	allenergies.append([x[1] for x in data[kpointfirstline+1:kpointfirstline+nrbands+1]])

Gnuplot.gp.GnuplotOpts.default_term = 'dumb'
g=Gnuplot.Gnuplot()
g('set terminal postscript; set nokey')
g('set output "bandstructure.ps"')
d = [Gnuplot.Data([x[i] for x in allenergies], with='lines') for i in range(nrbands)]
g.plot(*d)

#writer = csv.writer(open('entable','w'), delimiter="\t")
#writer.writerows(allenergies)

