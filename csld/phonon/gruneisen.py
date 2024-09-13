import subprocess
import numpy as np
import h5py
import scipy as sp

def get_total_grun(omega,grun,kweight,T):
    total = 0
    weight = 0
    nptk = omega.shape[0]
    nbands = omega.shape[1]
    hbar = sp.constants.hbar
    kB = sp.constants.Boltzmann
    if T == 0:
        total = 0
    else :
        for jj in range(nbands):
            for ii in range(nptk):
                x=hbar*omega[ii,jj]/(2.0*kB*T)
                dBE=(x/np.sinh(x))**2
                weight += dBE*kweight[ii]
                total += dBE*kweight[ii]*grun[ii,jj]
        total=total/weight
    return total

def gruneisen(primitive_fname,symprec,dim,mesh,temperatures,Cv,bulk_mod,vol) :
    # Anharmonic properties by internal call to phono3py (Gruneisen function)
    phono3py_cmd = 'phono3py -c '+primitive_fname+' --tolerance='+str(symprec)+' --dim="'+str(dim[0])+' '+str(dim[1])+' '+str(dim[2])+'"\
                 --fc2 --fc3 --sym-fc --br --gruneisen --mesh="'+str(mesh[0])+' '+str(mesh[1])+' '+str(mesh[2])+'" --ts="{0}"'.format(' '.join(str(T) for T in temperatures))
    print(phono3py_cmd)
    subprocess.call(phono3py_cmd, shell=True)
    grun = np.array(h5py.File('gruneisen.hdf5','r')['gruneisen'])
    omega = np.array(h5py.File('gruneisen.hdf5','r')['frequency'])*1e12*2*np.pi
    kweight = np.array(h5py.File('gruneisen.hdf5','r')['weight'])
    grun_tot = list()
    for temp in temperatures:
        grun_tot.append(get_total_grun(omega,grun,kweight,temp))
    grun_tot = np.nan_to_num(grun_tot)
    intCv = np.zeros(len(Cv))
    for tt in range(len(temperatures)):
        temp_max = temperatures[tt]
        if temp_max == 0:
            intCv[tt] = 0
        else:
            intCv[tt] = np.trapz(Cv[0:tt+1],temperatures[0:tt+1])  # integrate for cumulative Cv
    cte = Cv*grun_tot/(vol/10**30)/(bulk_mod*10**9)/3
    dLfrac = intCv*grun_tot/(vol/10**30)/(bulk_mod*10**9)/3
    cte = np.nan_to_num(cte)
    dLfrac = np.nan_to_num(dLfrac)

    return grun_tot, cte, dLfrac
