def listtostrangemolcasformat(mylist):
    vecchuncs=chunks(mylist,4)
    strangeformat=[[FloatToMolcasString(float(x)) for x in y] for y in vecchuncs]
    liste=["".join(x)+"\n" for x in strangeformat]
    return "".join(liste)

def chunks(l, n):
    #http://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks-in-python
    return [l[i:i+n] for i in range(0, len(l), n)]

def FloatToMolcasString(zahl):
    a='{:+018.12E}'.format(zahl)
    sign='0' if zahl>=0 else '-'
    exponent='E{:+03d}'.format(int(a[-3:])+1)
    zahl2='{:+015.12f}'.format(float(a[:15])/10)[2:]
    return sign+zahl2+exponent