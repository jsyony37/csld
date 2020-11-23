#!/usr/common/software/python/2.7-anaconda/bin/python
import os
import scipy
from scipy import array, linalg, dot
from csld.phonon import head
from numpy import *
from numpy.linalg import *
import copy
import numpy.linalg
from get_matcov import get_matcov
import multiprocessing as mp
import time

def mkdir(dir):
    if os.path.isdir(dir):
        os.system(str("rm"+" -r "+dir))
        os.system(str("mkdir"+" "+dir))
    else:
        os.system(str("mkdir"+" "+dir))


def get_ifc(file):
    fin=None
    fin=open(file,'r')
    fifc=fin.readlines()
    fin.close()
    natom=int(fifc[0])    
    ifc=array([[0.0 for i in range(3*natom)] for j in range(3*natom)])
    for i in range(natom):
        for j in range(natom):
            index=i*natom*4+j*4+1
            ifc[3*i+0, 3*j:3*j+3]=list(map(float, fifc[index+1].split()))
            ifc[3*i+1, 3*j:3*j+3]=list(map(float, fifc[index+2].split()))
            ifc[3*i+2, 3*j:3*j+3]=list(map(float, fifc[index+3].split()))
    return ifc
    
def get_dig(poscar,ifc):
    dyn=copy.deepcopy(ifc)
    #evc=copy.deepcopy(ifc)
    natom=poscar['natom']
    for i in range(natom):
        for j in range(natom):
            dyn[3*i:3*i+3, 3*j:3*j+3] = dyn[3*i:3*i+3, 3*j:3*j+3]/sqrt(poscar['mas'][i]*poscar['mas'][j])
    dyn=(dyn+transpose(dyn))/2.0
    #scipy.linalg.eig(dyn, evc, False, True, False,False,True)
    solution=numpy.linalg.eigh(dyn)
    return solution


def read_poscar(file,massall): #assuming SPOSCAR
    print(massall)
    fin=None
    fin=open(file, 'r')
    tmp=fin.readlines() #ftmp
    fin.close()
    poscar={}
    poscar['mas']=[] #done
    poscar['cor']=[] #done
    poscar['spe']=[]
    poscar['prefix']=tmp[0]
    poscar['scaling']=float(tmp[1])
    poscar['vec1']=list(map(float, tmp[2].split()))
    poscar['vec2']=list(map(float, tmp[3].split()))
    poscar['vec3']=list(map(float, tmp[4].split()))
    poscar['species']=tmp[5].split()
    poscar['specount']=list(map(int, tmp[6].split()))
    poscar['natom']=sum(poscar['specount'])
    poscar['coordinate']=tmp[7]
    for ia in range(poscar['natom']):
        poscar['cor'].append(array( list(map(float, (tmp[8+ia].split())[0:3] )) ))
    for ise in range(len(poscar['specount'])):
        iac=poscar['specount'][ise]
        for i in range(iac):
            poscar['mas'].append(massall[ise])
            poscar['spe'].append(poscar['species'][ise])
    return poscar

def write_poscar_cart(file, poscar, carposdis):
    print("Writing to "+str(file))
    dataout="System name \n"
    dataout+="1.0 \n"    
    dataout+=" ".join(map(str,poscar['vec1']))+"\n"
    dataout+=" ".join(map(str,poscar['vec2']))+"\n"
    dataout+=" ".join(map(str,poscar['vec3']))+"\n"
    dataout+=" ".join(map(str,poscar['species']))+"\n"
    dataout+=" ".join(map(str,poscar['specount']))+"\n"
    dataout+="C"+"\n"
    for pos in carposdis:
        dataout+=" ".join(map(str, pos))+"\n"
    ffout=None
    ffout=open(file, 'w')
    ffout.write(dataout)
    ffout.close()

def write_poscar_direct(file, poscar, carposdis):
    print("Writing to "+str(file))
    dataout="System name \n"
    dataout+="1.0 \n"
    dataout+=" ".join(map(str,poscar['vec1']))+"\n"
    dataout+=" ".join(map(str,poscar['vec2']))+"\n"
    dataout+=" ".join(map(str,poscar['vec3']))+"\n"
    #PRN uses no species
    dataout+=" ".join(map(str,poscar['species']))+"\n"
    dataout+=" ".join(map(str,poscar['specount']))+"\n"
    dataout+="Direct"+"\n"
    for pos in carposdis:
        dataout+=" ".join(map(str, pos))+"\n"
    ffout=None
    ffout=open(file, 'w')
    ffout.write(dataout)
    ffout.close()

def write_zero_forces(file, poscar, carposdis):
    print("Writing to "+str(file))
    dataout=""
    #dataout="System name \n"
    #dataout+="1.0 \n"
    #dataout+=" ".join(map(str,poscar['vec1']))+"\n"
    #dataout+=" ".join(map(str,poscar['vec2']))+"\n"
    #dataout+=" ".join(map(str,poscar['vec3']))+"\n"
    #PRN uses no species                                                                                                                                                           
    #dataout+=" ".join(map(str,poscar['species']))+"\n"                                                                                                                            
    #dataout+=" ".join(map(str,poscar['specount']))+"\n"
    #dataout+="Direct"+"\n"
    for pos in carposdis:
        dataout+="0.0  0.0  0.0\n"
    ffout=None
    ffout=open(file, 'w')
    ffout.write(dataout)
    ffout.close()


def dispWrap(dispAll,Lmatcov,natom,poscar,path,iconf):
    #dispAll = random.normal(mu, sigma, ndim)
    disp=(dot(Lmatcov, dispAll[iconf])).reshape((natom, 3)) 
    print("---------------------------------------------------") 
    print(norm(disp)/sqrt(len(disp)))
    latvec=array([poscar['vec1'],poscar['vec2'],poscar['vec3']])
    dirpos=poscar['cor']+dot(disp, inv(latvec))
    mkdir(path+"disp-"+str(iconf+1)) 
    write_poscar_direct(path+"disp-"+str(iconf+1)+"/POSCAR", poscar, dirpos)
#    write_poscar_direct(path+"disp-"+str(iconf+1)+"/POSCAR.direct", poscar, dirpos)
#    write_zero_forces(path+"disp-"+str(iconf+1)+"/force.txt", poscar, dirpos)
    
#-----------------------------------------------------------------------
def qcv_displace(masslist,temperature,nconfig,path,numprocess):
    #read POSCAR
    readPOSCAR=0
    if readPOSCAR > 1:
        tmp=get_yaml("./mesh.yaml")
        poscar=tmp[0]
        matevc=tmp[-1]
        print("******Check evc normalization: "+str(dot(matevc[:,0], matevc[:,0])))
    else:
        poscar=read_poscar(path+"SPOSCAR",masslist)  # mass for each species
#    print("POSCAR----------------------------------------------------------")
#    for item in poscar:
#        print("------"+str(item))
#        print(str(poscar[item]))
#    print("******Number of atoms: "+str(poscar['natom']))

    #read IFC
#    print("IFC-------------------------------------------------------------")
    ifc=get_ifc(path+"FORCE_CONSTANTS_2ND")
    #print ifc[0:3,3:6]

    #diagonalize Dynamical matrix
    [eig, evc]=get_dig(poscar,ifc)
    print("Frequency-------------------------------------------------------")
    print("******Number of eigenfrequencies: "+str(len(eig)))
    freq=[]
    unit2thz=98.1761/2/pi # eV / A^2 / au to THz
    unit2mev=unit2thz*4.135665538536  #THZ to meV
    unit=unit2mev
    unit=unit2thz
    isImag=0
    for tmp in eig:
        if tmp < 0:
            freq.append(-1*sqrt(-tmp)*unit)
            if sqrt(-tmp)*unit > 0.005: # check if it is acoustic modes
                isImag=isImag+1
        else:
            freq.append(sqrt(tmp)*unit)
    idx = array(freq).argsort()[::1]  
    head.writenumber(isImag, path+"isImag")
#    print("******Sort frequency index:")
#    print(idx)
#    print("******Unsorted frequencies (THz)")
#    print(array(freq))
    sorteig=array(freq)[idx]
    sortevc=evc[:,idx]
#    print("******Sorted frequencies (THz)")
#    print(sorteig)
#    print("Eigenvector-----------------------------------------------------")
    print("******Check evc normalization: ")
    print(dot(sortevc[:,0], sortevc[:,0]))
    head.write1dmat(list(sorteig), path+"frequencies")
    #print type(sortevc[:,0])
#    print(sortevc[:,0])

    # Construct quantum covariance matrix
    natom=poscar['natom']
    masses=poscar['mas']
    ndim=3*natom
    print("Number of atoms: "+str(natom))
    matcov=[[ 0.0 for i in range(ndim)] for j in range(ndim)]
    free_energy, matcov = get_matcov.pmatcov(temperature, natom, masses, path)
#    matcov=head.read2dmat(path+"matcov.dat")
#    print("Eigenvalues of covariance matrix: ")
#    print(eigvalsh(matcov))
    Lmatcov = scipy.linalg.cholesky(matcov,lower=True)
#    head.write2dmat(Lmatcov.tolist(), path+"Lmatcov")   
    latvec=array([poscar['vec1'],poscar['vec2'],poscar['vec3']]) * poscar['scaling']
#    print("Lattice vector: ")
#    print(latvec)
    #random seed to ensure the randoming sampling are same
#    random.seed(90001)
#    random.seed(100901)
    # pre-generate all random distributions
    mu = 0.0
    sigma = 1.0 # 0.5
    dispAll=[random.normal(mu, sigma, ndim) for iconf in range(nconfig)] # Gaussian random sampling
    stime=time.time()
    if numprocess == 1:
        results=[ dispWrap(dispAll,Lmatcov,natom,poscar,path,iconf) for iconf in range(nconfig) ]
    elif numprocess > 1:
        pool = mp.Pool(processes=numprocess)
        results=[ pool.apply_async(dispWrap, args=(dispAll,Lmatcov,natom,poscar,path,iconf)) for iconf in range(nconfig) ]
        pool.close()
        print("Pool closed")
        pool.join()
    else:
        raise ValueError("numprocess should be natural number")
    print("Time for generating displacements: "+str(time.time()-stime)+"\n")

    return free_energy


#make sure path exist
#--here is path=./pbte-images/
#prepare FORCE_CONSTANT_CSLD, FORCE_CONSTANT_CSLD_OLD and POSCAR-org under path 
#--FORCE_CONSTANT_CSLD and FORCE_CONSTANT_CSLD_OLD could be the same, if not, phonon frequency is the average one
#--FORCE_CONSTANT_CSLD and FORCE_CONSTANT_CSLD_OLD are all in Phonopy format
#--POSCAR-org is the supercell in line with FORCE_CONSTANT_CSLD
