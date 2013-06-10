import math

# XXX: MolecularOrbital.nr is set to str (normally int). is that good or bad?
def disentangle_mos(orb1, orb2):
    #the norm of the resulting states is not exactly 1 because the
    #occupations don't exactly add up to 2
    total_occupation = orb1.occupation + orb2.occupation
    
    orbplus = math.sqrt(orb1.occupation/2.)*orb1+math.sqrt(orb2.occupation/2.)*orb2
    orbminus = math.sqrt(orb1.occupation/2.)*orb1-math.sqrt(orb2.occupation/2.)*orb2

    orbplus.occupation = 1.
    orbminus.occupation = 1.
    
    orbplus.nr = '%i,%i,+'%(orb1.nr,orb2.nr)
    orbminus.nr = '%i,%i,-'%(orb1.nr,orb2.nr)
    
    return orbplus, orbminus

def disentangle_mo_set(orbs):
    # XXX: check reversed(orbs[-half_len:]) construction
    half_len = len(orbs)/2
    original = [(orb1, orb2) for orb1, orb2 in zip(orbs[:half_len],reversed(orbs[-half_len:]))]
    disentangled = [disentangle_mos(orb1, orb2) for orb1, orb2 in zip(orbs[:half_len],reversed(orbs[-half_len:]))]
    print [orb1.occupation+orb2.occupation for orb1, orb2 in zip(orbs[:half_len],reversed(orbs[-half_len:]))]
    return disentangled, original
