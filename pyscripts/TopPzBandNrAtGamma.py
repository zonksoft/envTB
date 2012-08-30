#!/usr/bin/env python
import procar
import sys

def read_command_line_arguments():
  if len(sys.argv) == 4:
    procarfilename = sys.argv[1]
    gnrwidth = int(sys.argv[2])
    wannier90filename = sys.argv[3]
  else:
    print("Usage: TopPzBandNrAtGamma.py procar_filename gnrwidth_rings wannier90.win_filename")
    sys.exit()    
  return procarfilename,gnrwidth,wannier90filename

if __name__ == "__main__":
  procarfilename,gnrwidth,wannier90filename=read_command_line_arguments()
  procarData=procar.ProcarData(procarfilename)

  nrbands,nrkpoints,nrions=procarData.info()
  chargedata=procarData.chargedata()
  energydata=procarData.energydata()

  #Nr of carbon atoms in a GNR of that width
  nrofpzbands=gnrwidth*2+2 
  #Charge data at Gamma point
  gammapointdata=chargedata[0] 
  #Sum the pz charge for a particular band at the gamma point over all ions. Do that for all bands.
  gammapointpzdata=[sum([ion[2] for ion in band]) for band in gammapointdata] 
  #Select band indices where there is pz charge at gamma
  selectpzbands=[i for i in range(len(gammapointpzdata)) if gammapointpzdata[i]>0.] 
  #Get band index of highest pz band at gamma point (index starting with 0, like always)
  highestgoodpzband=selectpzbands[nrofpzbands-1]
  #Energy of that band at gamma point 
  energyatgammaofhighestgoodpzband=energydata[0][highestgoodpzband]

  #append to wannier90 infile
  wannier90file = open(wannier90filename,'a')
  wannier90file.write('\nexclude_bands=' + str(highestgoodpzband+1+1) + '-' + str(nrbands)+'\n'+
                      'num_bands=' + str(highestgoodpzband+1)+'\n'+
                      'dis_win_max=' + str(energyatgammaofhighestgoodpzband+0.01)+'\n'
                     )
  wannier90file.close()
