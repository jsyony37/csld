#!/usr/bin/env python3

import sys
import os
import re
import logging
import glob
import atexit
import shutil
import copy
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_pdf import PdfPages

from csld.util.string_utils import str2arr
from csld.interface_vasp import Poscar
from csld.lattice_dynamics import init_ld_model
from csld.structure import SupercellStructure
from csld.symmetry_structure import SymmetrizedStructure
from csld.phonon.phonon import Phonon, NA_correction
from cssolve.csfit import csfit, predict_holdout
from csld.common_main import *
from csld.phonon.prn_get_gdisp import get_displacement
from gruneisen import gruneisen

import re
import logging
import glob
import numpy as np
from csld.util.string_utils import str2arr
from csld.interface_vasp import Poscar
from csld.structure import SupercellStructure
from csld.phonon.phonon import Phonon, NA_correction
from cssolve.csfit import csfit, predict_holdout
from csld.common_main import init_training


def fit_data(model, Amat, fval, setting, step, pdfout):
    """
    Fitting
    :param model: Lattice dynamics model
    :param Amat:
    :param fval:
    :param setting:
    :param step:
    :param pdfout:
    :return: optimal solution
    """
    if step <= 0:
       exit(0)
    print('\n')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!! COMPRESSIVE SENSING  !!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('\n')
    if step == 1:
        solutions = model.load_solution(setting['solution_in'],setting.getboolean('potential_coords_ijkl',False))
        if Amat is not None:
            err = [np.std(Amat.dot(solutions[i])-fval[:,0]) for i in range(solutions.shape[0])]
            ibest = np.argmin(err)
        else:
            ibest = 0
            if solutions.size <= 0:
                logging.error("ERROR: empty solution")
                exit(-1)
            if solutions.shape[0] > 1:
                logging.warning("More than 1 solutions found. Returning the first.")
        rel_err = 0
    elif step in [2, 3]:
        mulist = list(map(float, setting['mulist'].split()))
        submodels = [y.split() for x, y in setting.items() if re.match('submodel.*', x) is not None]
        submodels = [[x[0], list(map(int, x[1:]))] for x in submodels]
        print('submodels : ', submodels)
        uscale_list = list(map(float, setting['uscale_list'].split()))
        ldffscale_list = list(map(float, setting.get('ldffscale_list', '1').split()))
        knownsol = setting.get('solution_known', '')
        submodels = model.get_submodels(submodels, uscale_list, setting.getfloat('lr_pair_penalty',0.0),
                   ldffscale_list = ldffscale_list, knownsol=knownsol)
	
        ibest, solutions, rel_err = csfit(Amat, fval[:,0], 1, mulist,
                method=int(setting['method']),
                maxIter=int(setting['maxiter']),
                tol=float(setting['tolerance']),
                nSubset=int(setting['nsubset']),
                subsetsize=float(setting['subsetsize']),
                holdsize=float(setting['holdsize']),
                lbd=float(setting['lambda']),
# bcs options
                reweight=setting.getboolean('bcs_reweight', False),
                penalty=setting.get('bcs_penalty', 'arctan'),
                jcutoff=setting.getfloat('bcs_jcutoff',1E-7),
                sigma2=setting.getfloat('bcs_sigma2',-1.0),
                eta=setting.getfloat('bcs_eta',1E-3),
                fitf=setting.get('true_v_fit'),
                submodels=submodels, pdfout=pdfout)
        if step == 3:
            np.savetxt(setting['solution_out'], solutions)
            np.savetxt(setting['solution_out']+'_full', model.Cmat.T.dot(np.array(solutions)[:,:model.Cmat.shape[0]].T).T)
    else:
        print("ERROR: Unknown fit_step: ", step)
        exit(-1)
    if model.ldff is not None:
        model.ldff.plot_pairPES(solutions)
    print("+ Fitting done. Best solution", ibest)
    return ibest, solutions, rel_err


def save_pot(model, sol, setting, step, phonon):
    """
    :param model: the LD model
    :param sol the optimal solution vector
    :param settings:
    :return:
    """
    if step == 0:
        return
    print('\n')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!! SAVING FC AND POTENTIALS !!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('\n')
    scs = [y.split() for x, y in setting.items() if re.match(r'save_pot_cell.*', x)]
    combine= setting.getboolean('combine_improper', True)
    for i in scs:
        cell = np.array(list(map(int, i[1:])), dtype=int).reshape(3,3)
        model.save_fct(sol, i[0], cell, combine_improper=combine)
    if len(scs)>0:
        print("  + FCT saved to %d supercell(s)" % (len(scs)))

    # export_shengbte should be removed to disable (default), or "Nx Ny Nz 2 3 4 separated by space)"
    if 'export_shengbte' in setting.keys():
        numbers= list(map(int, setting['export_shengbte'].split()))
        orders = numbers[3:]
        use_old = setting.getboolean('original_shengbte_format', False)
        for ord in orders:
            if ord == 2:
            # note solution sol should already have been passed to phonon
                sc = SupercellStructure.from_scmat(model.prim, np.diag(numbers[:3]))
                if use_old:
                    phonon.export_hessian_forshengbte_original(sc)
                else:
                    phonon.export_hessian_forshengbte(sc)
                print('Dim after export_hessian : ', phonon.dim)
            elif ord in [3,4]:
                if use_old:
                    model.save_fcshengbte(sol, ord)
            elif ord in [3,4]:
                if use_old:
                    model.save_fcshengbte_original(sol, ord)
                else:
                    model.save_fcshengbte(sol, ord)
            else:
                print("cannot force constants of %d order"%(ord))
        print("  + shengbte format exported")
    print("+ FCT export done")


def predict(model, sols, setting, step):
    """
    :param model:
    :param sol:
    :param setting:
    :param step:
    :return:
    """
    if step <= 0:
        return
    elif step in [1, 2, 3]:
        Amat, fval = init_training(model, setting, step, delForce=0)
    else:
        print("ERROR: Unknown pred_step: ", step)
        exit(-1)

    print('\n')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!! PREDICTING USING FITTED RESULTS !!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('\n')

    errs = []
    for i in range(len(sols)):
        err = predict_holdout(Amat, fval[:, 0], sols[i])
        err[2:4]+= fval[:,2:].T
        errs.append(err[0])
        print("  sol# %d: err= (%.2f%%) %f" % (i, err[0], err[1]))
        np.savetxt("%s_%d"%(setting['fval_out'],i), np.transpose(err[2:4]))
        if setting.getboolean('save_force_prediction', True) and setting['corr_type']=='f':
            supercells= [y.split() for x, y in setting.items() if re.match('traindat.*', x)]
            left=0
            f_all= np.reshape(err[3],(-1,3))
            for sc in supercells:
                nA = Poscar.from_file(sc[0]).structure.num_sites
                for subs in sc[1:]:
                    for f in sorted(glob.glob(subs)):
                        np.savetxt(f+'/force.txt_predicted', f_all[left:left+nA])
                        left+=nA
    print("+ Prediction done")
    return np.argmin(errs)


def renormalization(model, settings, options, sol, temp, dLfrac, anh_order):
    """
    :param model: The LD model
    :param sol : The optimal solution vector from original fit
    :temp : the temperature at which the renormalization is performed
    :anh_order : the order of anharmonicity for renormalization (only 4th for now)
    :return:
    """
    print('\n')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!!! STARTING ANHARMONIC RENORMALIZATION @ ', temp, 'K !!!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('\n')

    path = str(temp)+'K/'
    primitive = settings['structure']['prim']
    nconfig = int(settings['renormalization']['nconfig'])
    nprocess = int(settings['renormalization']['nprocess'])
    mix_old = float(settings['renormalization']['mix_old'])
    conv_thresh = float(settings['renormalization']['conv_thresh'])
    settings['training']['traindat1'] = path+'SPOSCAR '+path+'disp*'
    nac = NA_correction.from_dict(settings['phonon'])
    etafac = settings['phonon'].getfloat('etafac', 8.0)    
    pdfout = 'plots-'+temp+'K.pdf'
    pdfout = PdfPages(pdfout.strip())
    atexit.register(upon_exit, pdfout)
    
    try:
        os.mkdir(path)
    except:
        pass
    if os.path.isfile('SPOSCAR'):
        pass
    else:
        raise ValueError('SPOSCAR not found!')
    fcfile = 'FORCE_CONSTANTS_2ND'
    shutil.copy(fcfile,fcfile+'_ORG')
    shutil.copy(fcfile+'_ORG',path+fcfile)
    shutil.copy(fcfile+'_ORG',path+fcfile+'_OLD')
    with open(primitive,'r') as f:
        lines = f.readlines()
        #lines[1] = str(float(lines[1])*(1+dLfrac))+' \n'
    with open(primitive+str(temp)+'K','w') as ff:
        ff.writelines(lines)
    with open('SPOSCAR','r') as f:
        lines = f.readlines()
        #lines[1] = str(float(lines[1])*(1+dLfrac))+' \n'
    with open(path+'SPOSCAR','w') as ff:
        ff.writelines(lines)
    prim = SymmetrizedStructure.init_structure(settings['structure'], options.symm_step, options.symm_prim, options.log_level, prim = primitive+str(temp)+'K')
    model = init_ld_model(prim, settings['model'], settings['LDFF'] if 'LDFF' in settings.sections() else {}, options.clus_step,
                          options.symC_step, options.ldff_step)
    print(prim.lattice.a)
    print(prim.lattice)
    
    count = 0
    while True:
        count += 1
        print('ITERATION ', count)
        # Diagnonalize IFC and Generate T-dependent atomic displacements (Yi)
        free_energy_diff = get_displacement(prim.atomic_masses,temp,nconfig,path,nprocess)
        shutil.copy(path+fcfile,path+fcfile+'_OLD')
        # Set-up T-dependent sensing matrix
        Amat_TD, fval_TD = init_training(model, settings['training'], step=2)
        nVal, nCorr = Amat_TD.shape

        if count == 1 : # Do this only once in the beginning
            sol = np.ones(nCorr)*sol
            sol_renorm = sol[:]
            param = model.get_params()
            print('params : ', param)
            start2 = 0
            for order in range(2):
                start2 += param[order]
            start4 = start2
            for order in range(2,anh_order):
                start4 += param[order]
            print('start2 : ', start2, ',  start4 : ', start4)
            sol2old = 0
            sol2orig = sol[start2:start2+param[2]]
            sol4 = sol[start4:start4+param[anh_order]]

        A2 = Amat_TD[:,start2:start2+param[2]].toarray()
        A4 = Amat_TD[:,start4:start4+param[anh_order]].toarray()
#        print('2nd order A dimensions : ', A2.shape)
#        print('2nd order sol dimensions :', sol2.shape)
#        print('4th order A dimensions : ', A4.shape)
#        print('4th order sol dimensions :', sol4.shape)

        ##### RENORMALIZE FC2 #####
        A2inv = np.linalg.pinv(A2) # Moore-Penrose pseudo-inverse...essentially a least-squares solver
        sol2new = A2inv.dot(A4.dot(sol4)) # least-squares solution
        sol_renorm[start2:start2+param[2]] = sol2orig + sol2old*mix_old + sol2new*(1-mix_old)
        print('Renormalized sol_renorm : \n', sol_renorm[start2:start2+param[2]])
        
        # Check relative difference in sol2new
        rel_chg = np.sum(abs(sol2new-sol2old)/abs(sol2new))/len(sol2new)
        rel_diff = np.sum(abs(sol2new)/abs(sol_renorm[start2:start2+param[2]]))/len(sol2new)
        print('Relative change in sol2new from sol2old is ', rel_chg)
        print('Relative difference from original sol2 is ', rel_diff)
        print('Relative change in free energy (meV/atom) is ', free_energy_diff)
        sol2old = sol2new[:]
        
        # Save renormalized FORCE_CONSTANTS_2ND
        phonon = Phonon(prim, model, sol_renorm, pdfout, NAC=nac, etafac=etafac)
        save_pot(model, sol_renorm, settings['export_potential'], 2, phonon)
        shutil.copy(fcfile,path+fcfile)

        # BREAK if relative difference in Free Energy is small
        if free_energy_diff < abs(conv_thresh) and count > 1:
            print('!!!!! Convergence Reached - Renormalization Done for ',str(temp),' K !!!!!')
            break

    sol_renorm = np.asarray(sol_renorm).reshape(1,nCorr)
    np.savetxt('solution_all_'+temp+'K', sol_renorm)
    phonon_step(model, prim, sol_renorm, settings['phonon'], temp, options.phonon_step, pdfout)    # Perform final phonon analysis
    shutil.move(fcfile+'_ORG',fcfile)
    for i in range(nconfig) :
        shutil.rmtree(path+'disp-'+str(i+1))

        
def phonon_step(model, prim, sols, setting, temp, step, pdfout):
    """
    :param model:
    :param prim
    :param setting:
    :param step:
    :return:
    """
    if step <= 0:
        return
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!! STARTING PHONON ANALYSIS !!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # set up NAC
    entries = [k for k, v in setting.items()]
    nac = NA_correction.from_dict(setting)
    unit = setting.get('unit', 'THz')
    etafac = setting.getfloat('etafac', 8.0)
    if step == 1:
        for i, sol in enumerate(sols):
            phonon = Phonon(prim, model, sol, pdfout, NAC=nac, etafac=etafac)
            cart=setting.getboolean("qpoint_cart")
            # logging.info('assuming '+('cartesian' if cart else 'fractional' )+' input q-points')
            # dispersion
            if 'wavevector' in entries:
#                kpts = 'Auto' if setting['wavevector']=='Auto' else str2arr(setting['wavevector'], shape=(-1, 3))
                kpts = setting['wavevector']
                eigE, kpts_line = phonon.get_dispersion(kpts, unit=unit, cart=cart, no_gamma=setting.getboolean("no_gamma",True))
                print('  + phonon dispersion generated')
            if 'eigen_wavevector' in entries:
                kpts = str2arr(setting['eigen_wavevector']).reshape((-1,3))
                phonon.get_eig_e_vec(kpts, unit=unit, cart=cart)
                print('  + eigenvectors exported')
            if 'dos_grid' in entries:
                ngrid = str2arr(setting['dos_grid'], int)
                ismear = setting.getint('ismear', -1)
                epsilon= setting.getfloat('epsilon')
                # logging.info('    DOS ismear=%d smearing width=%f' %(ismear, epsilon))
                pdos = setting.getboolean('pdos', False)
                dos=phonon.get_dos(ngrid, int(setting['nE_dos']), ismear, epsilon, temp, unit=unit, pdos=pdos, no_gamma=setting.getboolean("no_gamma",True))
                print('  + phonon DOS%s generated'% (' + partial DOS' if pdos else ''))
                if 'thermal_t_range' in entries:
                    if temp == 0 :
                        t_rng = str2arr(setting['thermal_t_range'])
                        t_rng = np.arange(*(t_rng.tolist()))
                    else:
                        t_rng = temp
                    thermal_dat = Phonon.calc_thermal_QHA(dos, t_rng, setting['thermal_out'])
                    print('  + harmonic phonon thermal properties calculated')

            if 'debye_t_qfrac' in entries:
                d_T_grid=list(map(int,setting.get('debye_t_v_intgrid','20 20').split()))
                d_T_qfac=list(map(float,setting['debye_t_qfrac'].split()))
                d_T, d_v =phonon.debye_T(d_T_grid, *d_T_qfac)
                print("  + Debye T (K), v (m/s) at %s fractional q point =%.5f %.2f"%(d_T_qfac, d_T, d_v))
                print("  for each acoustic branch", phonon.debye_T(d_T_grid, *d_T_qfac, False)[0])

            if 'supercell' in entries:
                sc = SupercellStructure.from_scmat(prim, str2arr(setting['supercell'],int, (3,3)))
                if 'snapshot_t' in entries:
                    phonon.supercell_snapshot(sc, float(setting['snapshot_t']), int(setting.get('snapshot_n', '10')))
                    print('  + phonon thermalized snapshots exported')
                # frozen phonons
                if 'modes' in entries:
                    if 'reference_structure' in entries:
                        posRef = Poscar.from_file(setting['reference_structure']).structure
                        logging.info('Output structure will follow order of atoms in '+setting['reference_structure'])
                    else:
                        posRef = None
                    if 'mode_amplitude' not in entries:
                        logging.error('Need "mode_amplitude" e.g. "0.1 0.2" in settings to export phonon modes.')
                        exit(0)
                    phonon.export_phononmode(sc, str2arr(setting['mode_amplitude']),
                                             str2arr(setting['modes'],shape=(-1,4)), cart=cart, posRef=posRef)
                    print('   frozen phonon modes exported')
                # covariance matrix
                if 'covariance_matrix_t' in entries:
                    np.savetxt('covariance_matrix.out', phonon.covariance_matrix_in_supercell(sc, float(setting['covariance_matrix_t'])))
                # NOTE: moved invocation of this function to [export_potential] export_shengbte=...
                #if bool(setting.getboolean('fc2shengbte',False)):
                #    phonon.export_hessian_forshengbte(sc)
                #    print('   simplified force constants for shengbte exported')

    else:
        print("ERROR: Unknown phonon_step: ", step)
        exit(-1)

    print("+ Phonon done")

    return phonon, kpts_line, thermal_dat


def anharmonic(prim, phonon, kpts_line, thermal_dat, settings):
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print('!!!!! STARTING ANHARMONIC ANALYSIS !!!!!')
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    # Grueneisen parameter
    ngrid = str2arr(settings['phonon']['dos_grid'], int)
    kgrid_unif = np.mgrid[0:1:1./ngrid[0], 0:1:1./ngrid[1], 0:1:1./ngrid[2]].transpose((1,2,3,0)).reshape(-1,3)
#    kgrid_unif, wt = prim.syminfo.get_ir_reciprocal_mesh(ngrid)
#    print('kgrid_unif : ',type(kgrid_unif),kgrid_unif.shape,'\n',kgrid_unif)
#    kgrid_unif_cart = kgrid_unif.dot(prim.lattice.reciprocal_lattice.matrix)
    kgrid_unif_cart = prim.reciprocal_lattice.get_cartesian_coords(kgrid_unif)
    np.savetxt('kgrid_unif',kgrid_unif)
    np.savetxt('kgrid_unif_cart',kgrid_unif_cart)
    kpts_line_cart = np.around(kpts_line,4)
#    kpts_line = prim.reciprocal_lattice.get_fractional_coords(kpts_line_cart)
#    kgrid_line = np.transpose(np.linalg.inv(np.transpose(prim.lattice.reciprocal_lattice.matrix)).dot(np.transpose(kpts_line)))
#    np.savetxt('kpts_line',kpts_line)
#    print('kpts_line : ',type(kpts_line),kpts_line.shape,'\n',kpts_line)
    np.savetxt('kpts_line',kpts_line)
    np.savetxt('kpts_line_cart',kpts_line_cart)
    print("Diagonalizing for eigenvalues and eigenvectors on uniform mesh...")
    eigE, eigV = phonon.get_eig_e_vec(kgrid_unif, unit='THz', cart=False)
    eigE = np.transpose(eigE)
    eigE_line, eigV_line = phonon.get_eig_e_vec(kpts_line, unit='THz', cart=False)
    eigE_line = np.transpose(eigE_line)
    eigV = np.transpose(eigV,(2,1,0))
    eigV_line = np.transpose(eigV_line,(2,1,0))
    np.savetxt("eigE.txt", eigE,delimiter='  ')
    np.savetxt("eigE_line.txt", eigE_line,delimiter='  ')
    np.savetxt('eigV-Gamma.txt',np.real(eigV[0,:,:]).round(4),delimiter='  ',fmt='%.5f')
    print('  + eigenvalues and eigenvectors obtained')
    try:
        fc3 = open('FORCE_CONSTANTS_3RD','r')
    except:
        print("+ No FORCE_CONSTANTS_3RD")
        return 0, 0
    Ntri = int(fc3.readline())
    lattvec = np.transpose(prim.lattice.matrix)
    print('Lattice Vectors :\n',lattvec)
    Phi,R_j,R_k,Index_i,Index_j,Index_k = gruneisen.read3fc(lattvec,Ntri)
    print('  + FORCE_CONSTANTS_3RD read ')
    masses = np.asarray(prim.atomic_masses)
    Index_i = np.transpose(Index_i.astype(int))
    Index_j = np.transpose(Index_j.astype(int))
    Index_k = np.transpose(Index_k.astype(int))
    cart_coord = np.transpose(prim.cart_coords)
#    rlattvec = np.transpose(prim.lattice.reciprocal_lattice.matrix)
#    print('Rlattvec : ',type(rlattvec),rlattvec.shape,'\n',rlattvec)  
    print('Cartesian Coord. : ',type(cart_coord),cart_coord.shape,'\n',cart_coord)#
    print('Eigenvalues at Gamma : ',type(eigE),eigE.shape,'\n',eigE[0,:])
    print('Eigenvectors at Gamma : ',type(eigV),eigV.shape,'\n',eigV[0,:,:])
    print('masses : ',type(masses),masses.shape,'\n',masses)
    print('Index_i : ',type(Index_i),Index_i.shape)
    print('R_j : ',type(R_j),R_j.shape,R_j)
    # omega must be multiplied by 2pi to be converted to rad*THz (angular frequency)  
    grun = gruneisen.mode_grun(eigE*2*np.pi,eigV,Phi,R_j,R_k,Index_i,Index_j,Index_k,cart_coord,masses,kgrid_unif_cart)#,rlattvec)
    np.savetxt('gruneisen_mode.txt',grun)
    t_rng = str2arr(settings['phonon']['thermal_t_range']).astype(int)
    t_rng = range(t_rng[0],t_rng[1],t_rng[2])
    total_grun = np.zeros(len(t_rng))
    for tt in range(len(t_rng)):
        total_grun[tt] = gruneisen.total_grun(eigE,grun,t_rng[tt])
    total_grun = np.nan_to_num(total_grun)
    print('  + The total Gruneisen parameter : \n', total_grun)
    np.savetxt('gruneisen_total.txt',total_grun)
    grun_line = gruneisen.mode_grun(eigE_line*2*np.pi,eigV_line,Phi,R_j,R_k,Index_i,Index_j,Index_k,cart_coord,masses,kpts_line_cart)#,rlattvec)
    np.savetxt('gruneisen_mode_line.txt',grun_line)

    # Thermal expansion (need bulk modulus input)
    bulk_mod = float(settings['anharmonic']['bulk_mod'])*10**9 # GPa to SI
    vol = prim.lattice.volume/10**30 # A^3 to SI
    Cv = thermal_dat[:,4]*1.3806/10**23 # unit conversion to SI 
    intCv = np.zeros(len(t_rng))
    for tt in range(len(t_rng)):
        temp_max = thermal_dat[tt,0]
        if temp_max == 0:
            intCv[tt] = 0
        else:
            intCv[tt] = np.trapz(Cv[range(tt)],thermal_dat[range(tt),0]) # integrate for cumulative Cv up to temp_max
    print('  + Heat capacity (J/K) : \n', Cv)
    cte = np.asarray(list(zip(t_rng,Cv*total_grun/vol/bulk_mod/3)))
    dLfrac = np.asarray(list(zip(t_rng,intCv*total_grun/vol/bulk_mod/3)))
    cte = np.nan_to_num(cte)
    dLfrac = np.nan_to_num(dLfrac)
    print('  + Coefficient of linear thermal expansion ( /K) : \n', cte)
    print('  + Linear thermal expansion fraction int(CTE*T*dT) : \n', dLfrac)

    return cte, dLfrac
