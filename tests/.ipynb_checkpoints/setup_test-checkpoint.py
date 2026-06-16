"""Shared test harness for building configured Vlasov-Maxwell / Vlasov-Poisson systems.

Provides helper routines (e.g. constructing the v^2 multiplication MPO and
assembling configured :class:`VlasovMaxwell` / :class:`VlasovPoisson` systems)
used by the Vlasov test drivers. Not a standalone test itself.
"""
import os, sys, pickle, time, glob
import numpy as np

sys.path.append('../')

from setup_.configs import *
import setup_.helper as helper_test
import helper_quimb as helper

from field import Field, ScalarField
from pde_vlasovEM import VlasovMaxwell
from pde_vlasovES import VlasovPoisson
from gridTN import GridTN


def get_v2_mpo(grid, axes, compress=True, compress_opts=None):
    """ get MPO that computes v^2 * f(x,v)
    """
    tot_mpo: 'GridTN' = None
    for ax in axes:
        print('ax basis', ax.basis)
        v2_mpo = grid.get_xmultiply_mpo(x_axes=[ax], x_power=2)
        if tot_mpo is None:
            tot_mpo = v2_mpo.copy()
        else:
            tot_mpo = tot_mpo.add(v2_mpo, inplace=True)

    print('tot mpo v2', tot_mpo.max_bond())
    if compress:
        tot_mpo.compress(inplace=True, compress_opts=compress_opts)
        print('compressed', tot_mpo.max_bond())

    return tot_mpo



def load_data_EM(VP_test,
                 init_fe_field: 'ScalarField', init_fi_field: 'ScalarField',
                 init_E_field: 'Field', init_B_field: 'Field',
                 init_phi_field: 'ScalarField', init_psi_field: 'ScalarField',
                 fdir: str = '', sdir: str = '', extra_str:str = '', save_every_nt = 250,
                 restart_nt=None,
                 load_data=True, restart=True, restart_from_T=0, other_data:Iterable=None, only_elc=False):

    ### get file naming string
    fstr = VP_test.get_fstr(extra_str=extra_str)
    print('filename', fstr)

    ### load data ###
    try:
        if not load_data:  raise IOError
        init_fe_field = init_fe_field.reload_data(fdir + 'fe_' + fstr)
        init_fi_field = init_fi_field.reload_data(fdir + 'fi_' + fstr) if not only_elc else None
        init_E_field = init_E_field.reload_data(fdir + 'E_' + fstr) if not only_elc else None
        init_B_field = init_B_field.reload_data(fdir + 'B_' + fstr) if not only_elc else None
        try:
            init_phi_field = init_phi_field.reload_data(fdir + 'phi_' + fstr)
            init_psi_field = init_psi_field.reload_data(fdir + 'psi_' + fstr)
        except IOError:
            pass

        max_bond_fe = pickle.load(open(fdir + 'maxD_fe_' + fstr + '.pkl', 'rb'))
        max_bond_fi = pickle.load(open(fdir + 'maxD_fi_' + fstr + '.pkl', 'rb')) if not only_elc else None
        max_bond_E = pickle.load(open(fdir + 'maxD_E_' + fstr + '.pkl', 'rb')) if not only_elc else None
        max_bond_B = pickle.load(open(fdir + 'maxD_B_' + fstr + '.pkl', 'rb')) if not only_elc else None
        try:
            max_bond_phi = pickle.load(open(fdir + 'maxD_phi_' + fstr + '.pkl', 'rb'))
            max_bond_psi = pickle.load(open(fdir + 'maxD_psi_' + fstr + '.pkl', 'rb'))
        except IOError:
            max_bond_phi = [np.nan]
            max_bond_psi = [np.nan]
       
        nrg_fe_ts = pickle.load(open(fdir + 'nrg_e_' + fstr + '.pkl', 'rb'))
        nrg_fi_ts = pickle.load(open(fdir + 'nrg_i_' + fstr + '.pkl', 'rb')) if not only_elc else None
        nrg_E_ts = pickle.load(open(fdir + 'nrg_E_' + fstr + '.pkl', 'rb')) if not only_elc else None
        nrg_B_ts = pickle.load(open(fdir + 'nrg_B_' + fstr + '.pkl', 'rb')) if not only_elc else None
        ts = pickle.load(open(fdir + 'ts_' + fstr + '.pkl', 'rb'))

        tmp_data = None
        if other_data is not None:
            tmp_data = []
            for data_name in other_data:
                tmp_data += [pickle.load(open(fdir + str(data_name) + fstr + '.pkl', 'rb'))]

        print('loaded data')

    except(IOError, EOFError):
        if restart_from_T is not None:
            restart_fstr = fstr
            init_fe_field = init_fe_field.reload_data(fdir + 'fe_' + restart_fstr)
            init_fi_field = init_fi_field.reload_data(fdir + 'fi_' + restart_fstr) if not only_elc else None
            init_E_field = init_E_field.reload_data(fdir + 'E_' + restart_fstr) if not only_elc else None
            init_B_field = init_B_field.reload_data(fdir + 'B_' + restart_fstr) if not only_elc else None
            try:
                init_phi_field = init_phi_field.reload_data(fdir + 'phi_' + restart_fstr)
                init_psi_field = init_psi_field.reload_data(fdir + 'psi_' + restart_fstr)
            except IOError:
                pass

            max_bond_fe = pickle.load(open(fdir + 'maxD_fe_' + restart_fstr + '.pkl', 'rb'))
            max_bond_fi = pickle.load(open(fdir + 'maxD_fi_' + restart_fstr + '.pkl', 'rb')) if not only_elc else None
            max_bond_E = pickle.load(open(fdir + 'maxD_E_' + restart_fstr + '.pkl', 'rb')) if not only_elc else None
            max_bond_B = pickle.load(open(fdir + 'maxD_B_' + restart_fstr + '.pkl', 'rb')) if not only_elc else None
            try:
                max_bond_psi = pickle.load(open(fdir + 'maxD_psi_' + restart_fstr + '.pkl', 'rb'))
                max_bond_phi = pickle.load(open(fdir + 'maxD_phi_' + restart_fstr + '.pkl', 'rb'))
            except IOError:
                max_bond_phi = [np.nan]
                max_bond_psi = [np.nan]

            nrg_fe_ts = pickle.load(open(fdir + 'nrg_e_' + fstr + '.pkl', 'rb'))
            nrg_fi_ts = pickle.load(open(fdir + 'nrg_i_' + fstr + '.pkl', 'rb')) if not only_elc else None
            nrg_E_ts = pickle.load(open(fdir + 'nrg_E_' + fstr + '.pkl', 'rb')) if not only_elc else None
            nrg_B_ts = pickle.load(open(fdir + 'nrg_B_' + fstr + '.pkl', 'rb')) if not only_elc else None
            ts = pickle.load(open(fdir + 'ts_' + fstr + '.pkl', 'rb'))

            tmp_data = None
            if other_data is not None:
                tmp_data = []
                for data_name in other_data:
                    tmp_data += [pickle.load(open(fdir + f'{data_name}_' + fstr + '.pkl', 'rb'))]

            print('restarting from ', restart_fstr)
            ts = list(ts)
            nt = len(ts) - 1
        else:
            if not restart:  raise ValueError

            re_fstr = fstr
            print(re_fstr)

            restart_fstrs = glob.glob(sdir + 'restart/fe_' + re_fstr + '-nt*.pkl')
            print(sdir + 'restart/')
            print('restart?', len(restart_fstrs))
            nts = []
            for f in restart_fstrs:
                nt_ind = f[-20:].find('-nt')
                nt_str = f[-20:][nt_ind + 3:-4]
                # print(f[-20:], nt_ind, nt_str)
                nts += [int(nt_str)]

            if restart_nt is None:
                nt = max(nts)
                print('max nt', nt)

                ### check that energy matches this
                nrg_fe_ts = pickle.load(open(sdir + 'restart/nrg_e_' + re_fstr + f'.pkl', 'rb'))
                nt_ = len(nrg_fe_ts) - 1
                print('nrg len nt', nt_)

                nt = min(nt_,nt)

            else:
                nt = restart_nt
                print('use nt', nt)
                print('max nt', max(nts))

            nts.remove(nt)

            try:
                # raise NameError
                for nt_ in nts:  ## remove old restart files
                    # if nt_ > nt or nt_ % save_every_nt != 0:
                    if nt_ % save_every_nt != 0:
                        print('removing restart', nt_)
                        restart_files = glob.glob(sdir + 'restart/*' + re_fstr + f'-nt{nt_}.pkl')
                        for f in restart_files:  os.remove(f)
            except NameError:
                pass

            init_fe_field = init_fe_field.reload_data(sdir + 'restart/fe_' + re_fstr + f'-nt{nt}')
            init_fi_field = init_fi_field.reload_data(sdir + 'restart/fi_' + re_fstr + f'-nt{nt}') if not only_elc else []
            init_E_field = init_E_field.reload_data(sdir + 'restart/E_' + re_fstr + f'-nt{nt}') if not only_elc else []
            init_B_field = init_B_field.reload_data(sdir + 'restart/B_' + re_fstr + f'-nt{nt}') if not only_elc else []
            if init_phi_field is not None:
                try:
                    init_phi_field = init_phi_field.reload_data(sdir + 'restart/phi_' + re_fstr + f'-nt{nt}')
                    init_psi_field = init_psi_field.reload_data(sdir + 'restart/psi_' + re_fstr + f'-nt{nt}')
                except IOError:
                    print('phi, psi not found')
                    pass

            max_bond_fe = pickle.load(open(sdir + 'restart/maxD_fe_' + re_fstr + f'.pkl', 'rb'))
            max_bond_fi = pickle.load(open(sdir + 'restart/maxD_fi_' + re_fstr + f'.pkl', 'rb')) if not only_elc else []
            max_bond_E = pickle.load(open(sdir + 'restart/maxD_E_' + re_fstr + f'.pkl', 'rb')) if not only_elc else []
            max_bond_B = pickle.load(open(sdir + 'restart/maxD_B_' + re_fstr + f'.pkl', 'rb')) if not only_elc else []
            try:
                max_bond_phi = pickle.load(open(sdir + 'restart/maxD_phi_' + re_fstr + f'.pkl', 'rb'))
                max_bond_psi = pickle.load(open(sdir + 'restart/maxD_psi_' + re_fstr + f'.pkl', 'rb'))
            except IOError:
                max_bond_phi = [np.nan]
                max_bond_psi = [np.nan]
            nrg_fe_ts = pickle.load(open(sdir + 'restart/nrg_e_' + re_fstr + f'.pkl', 'rb'))
            nrg_fi_ts = pickle.load(open(sdir + 'restart/nrg_i_' + re_fstr + f'.pkl', 'rb')) if not only_elc else []
            nrg_E_ts = pickle.load(open(sdir + 'restart/nrg_E_' + re_fstr + f'.pkl', 'rb')) if not only_elc else []
            nrg_B_ts = pickle.load(open(sdir + 'restart/nrg_B_' + re_fstr + f'.pkl', 'rb')) if not only_elc else []
            ts = pickle.load(open(sdir + 'restart/ts_' + re_fstr + f'.pkl', 'rb'))
            ts = list(ts)
            print('loaded', nt, len(nrg_fe_ts))

            tmp_data = None
            if other_data is not None:
                tmp_data = []
                for data_name in other_data:
                    tmp_data += [pickle.load(open(sdir + f'restart/{data_name}_' + re_fstr + '.pkl', 'rb'))]


        max_bond_fe = max_bond_fe[:nt + 1]
        max_bond_fi = max_bond_fi[:nt + 1]
        max_bond_E = max_bond_E[:nt + 1]
        max_bond_B = max_bond_B[:nt + 1]
        max_bond_phi = max_bond_phi[:nt + 1]
        max_bond_psi = max_bond_psi[:nt + 1]
        nrg_fe_ts = nrg_fe_ts[:nt + 1]
        nrg_fi_ts = nrg_fi_ts[:nt + 1]
        nrg_E_ts = nrg_E_ts[:nt + 1]
        nrg_B_ts = nrg_B_ts[:nt + 1]
        ts = ts[:nt + 1]

        if tmp_data is not None:
            for k in range(len(other_data)):
                tmp_data[k] = tmp_data[k][:nt + 1]

        print('loaded restart', nt)

    if tmp_data is not None:
        if only_elc:
            return init_fe_field, ts, nrg_fe_ts, max_bond_fe, *tmp_data

        return (init_fe_field, init_fi_field, init_E_field, init_B_field, init_phi_field, init_psi_field,
                ts, nrg_fe_ts, nrg_fi_ts, nrg_E_ts, nrg_B_ts,
                max_bond_fe, max_bond_fi, max_bond_E, max_bond_B, max_bond_phi, max_bond_psi, *tmp_data)
    else:
        if only_elc:
            return init_fe_field, ts, nrg_fe_ts, max_bond_fe

        return (init_fe_field, init_fi_field, init_E_field, init_B_field, init_phi_field, init_psi_field,
                ts, nrg_fe_ts, nrg_fi_ts, nrg_E_ts, nrg_B_ts,
                max_bond_fe, max_bond_fi, max_bond_E, max_bond_B, max_bond_phi, max_bond_psi,)


def save_data_EM(VP_test, vm_sys, fdir, extra_str='', add_fstr='',
                 ts=None, nrg_fe_ts=None, nrg_fi_ts=None, nrg_E_ts=None, nrg_B_ts=None,
                 max_bond_fe=None, max_bond_fi=None, max_bond_E=None, max_bond_B=None,
                 max_bond_phi=None, max_bond_psi=None, other_data: dict['str',Any]=None, only_elc=False ):
    """ save data EM
    """
    fstr = VP_test.get_fstr(extra_str=extra_str)
    f_fstr = fstr + add_fstr

    vm_sys.fe.save_data(fdir + 'fe_' + f_fstr)
    if not only_elc:
        vm_sys.fi.save_data(fdir + 'fi_' + f_fstr)
        vm_sys.E.save_data(fdir + 'E_' + f_fstr)
        vm_sys.B.save_data(fdir + 'B_' + f_fstr)
        if vm_sys.phi is not None:
            vm_sys.phi.save_data(fdir + 'phi_' + f_fstr)
        if vm_sys.psi is not None:
            vm_sys.psi.save_data(fdir + 'psi_' + f_fstr)

    if not np.isnan(max_bond_fe[-1]):
        pickle.dump(max_bond_fe, open(fdir + 'maxD_fe_' + fstr + '.pkl', 'wb'))
    if not only_elc:
        if not np.isnan(max_bond_fi[-1]):
            pickle.dump(max_bond_fi, open(fdir + 'maxD_fi_' + fstr + '.pkl', 'wb'))
        if not all(np.isnan(max_bond_E[-1])):
            pickle.dump(max_bond_E, open(fdir + 'maxD_E_' + fstr + '.pkl', 'wb'))
        if not all(np.isnan(max_bond_B[-1])):
            pickle.dump(max_bond_B, open(fdir + 'maxD_B_' + fstr + '.pkl', 'wb'))
        if max_bond_phi is not None and not np.isnan(max_bond_phi[-1]):
            pickle.dump(max_bond_phi, open(fdir + 'maxD_phi_' + fstr + '.pkl', 'wb'))
        if max_bond_psi is not None and not np.isnan(max_bond_psi[-1]):
            pickle.dump(max_bond_psi, open(fdir + 'maxD_psi_' + fstr + '.pkl', 'wb'))

    pickle.dump(nrg_fe_ts, open(fdir + 'nrg_e_' + fstr + '.pkl', 'wb'))
    if not only_elc:
        pickle.dump(nrg_fi_ts, open(fdir + 'nrg_i_' + fstr + '.pkl', 'wb'))
        pickle.dump(nrg_E_ts, open(fdir + 'nrg_E_' + fstr + '.pkl', 'wb'))
        pickle.dump(nrg_B_ts, open(fdir + 'nrg_B_' + fstr + '.pkl', 'wb'))
    pickle.dump(ts, open(fdir + 'ts_' + fstr + '.pkl', 'wb'))

    if other_data is not None:
        for k, v in other_data.items():
            pickle.dump(v, open(fdir + str(k) + '_' + fstr + '.pkl', 'wb'))


def load_data_Maxwell(fstr,
                      init_E_field: 'Field', init_B_field: 'Field',
                      init_phi_field: 'ScalarField'=None, init_psi_field: 'ScalarField'=None,
                      fdir: str = '', sdir: str = '', save_every_nt=250,
                      restart_nt=None,
                      load_data=True, restart=True, restart_from_T=0, other_data: Iterable = None):
    ### get file naming string
    # fstr = VP_test.get_fstr(extra_str=extra_str)
    print('filename', fstr)

    ### load data ###
    try:
        if not load_data:  raise IOError
        init_E_field = init_E_field.reload_data(fdir + 'E_' + fstr)
        init_B_field = init_B_field.reload_data(fdir + 'B_' + fstr)
        try:
            init_phi_field = init_phi_field.reload_data(fdir + 'phi_' + fstr)
            init_psi_field = init_psi_field.reload_data(fdir + 'psi_' + fstr)
        except (AttributeError, IOError):
            pass

        max_bond_E = pickle.load(open(fdir + 'maxD_E_' + fstr + '.pkl', 'rb'))
        max_bond_B = pickle.load(open(fdir + 'maxD_B_' + fstr + '.pkl', 'rb'))
        try:
            max_bond_phi = pickle.load(open(fdir + 'maxD_phi_' + fstr + '.pkl', 'rb'))
            max_bond_psi = pickle.load(open(fdir + 'maxD_psi_' + fstr + '.pkl', 'rb'))
        except IOError:
            max_bond_phi = [np.nan]
            max_bond_psi = [np.nan]

        nrg_E_ts = pickle.load(open(fdir + 'nrg_E_' + fstr + '.pkl', 'rb'))
        nrg_B_ts = pickle.load(open(fdir + 'nrg_B_' + fstr + '.pkl', 'rb'))
        ts = pickle.load(open(fdir + 'ts_' + fstr + '.pkl', 'rb'))

        tmp_data = None
        if other_data is not None:
            tmp_data = []
            for data_name in other_data:
                tmp_data += [pickle.load(open(fdir + str(data_name) + fstr + '.pkl', 'rb'))]

        print('loaded data')

    except(IOError, EOFError):
        if restart_from_T is not None:
            restart_fstr = fstr
            init_E_field = init_E_field.reload_data(fdir + 'E_' + restart_fstr)
            init_B_field = init_B_field.reload_data(fdir + 'B_' + restart_fstr)
            try:
                init_phi_field = init_phi_field.reload_data(fdir + 'phi_' + restart_fstr)
                init_psi_field = init_psi_field.reload_data(fdir + 'psi_' + restart_fstr)
            except IOError:
                pass

            max_bond_E = pickle.load(open(fdir + 'maxD_E_' + restart_fstr + '.pkl', 'rb'))
            max_bond_B = pickle.load(open(fdir + 'maxD_B_' + restart_fstr + '.pkl', 'rb'))
            try:
                max_bond_psi = pickle.load(open(fdir + 'maxD_psi_' + restart_fstr + '.pkl', 'rb'))
                max_bond_phi = pickle.load(open(fdir + 'maxD_phi_' + restart_fstr + '.pkl', 'rb'))
            except IOError:
                max_bond_phi = [np.nan]
                max_bond_psi = [np.nan]

            nrg_E_ts = pickle.load(open(fdir + 'nrg_E_' + fstr + '.pkl', 'rb'))
            nrg_B_ts = pickle.load(open(fdir + 'nrg_B_' + fstr + '.pkl', 'rb'))
            ts = pickle.load(open(fdir + 'ts_' + fstr + '.pkl', 'rb'))

            tmp_data = None
            if other_data is not None:
                tmp_data = []
                for data_name in other_data:
                    tmp_data += [pickle.load(open(fdir + f'{data_name}_' + fstr + '.pkl', 'rb'))]

            print('restarting from ', restart_fstr)
            ts = list(ts)
            nt = len(ts) - 1
        else:
            if not restart:  raise ValueError

            re_fstr = fstr
            print(re_fstr)

            restart_fstrs = glob.glob(sdir + 'restart/E_' + re_fstr + '-nt*.pkl')
            print('sdir', sdir + 'restart/')
            print('restart?', len(restart_fstrs))
            nts = []
            for f in restart_fstrs:
                nt_ind = f[-20:].find('-nt')
                nt_str = f[-20:][nt_ind + 3:-6]
                # print(f[-20:], nt_ind, nt_str)
                nts += [int(nt_str)]

            if restart_nt is None:
                nt = max(nts)
                print('max nt', nt)

                ### check that energy matches this
                nrg_fe_ts = pickle.load(open(sdir + 'restart/ts_' + re_fstr + f'.pkl', 'rb'))
                nt_ = len(nrg_fe_ts) - 1
                print('nrg len nt', nt_)

                nt = min(nt_, nt)

            else:
                nt = restart_nt
                print('use nt', nt)
                print('max nt', max(nts))

            nts.remove(nt)

            try:
                # raise NameError
                for nt_ in nts:  ## remove old restart files
                    # if nt_ > nt or nt_ % save_every_nt != 0:
                    if nt_ % save_every_nt != 0:
                        print('removing restart', nt_)
                        restart_files = glob.glob(sdir + 'restart/*' + re_fstr + f'-nt{nt_}.pkl')
                        for f in restart_files:  os.remove(f)
            except NameError:
                pass

            init_E_field = init_E_field.reload_data(sdir + 'restart/E_' + re_fstr + f'-nt{nt}')
            init_B_field = init_B_field.reload_data(sdir + 'restart/B_' + re_fstr + f'-nt{nt}')
            if init_phi_field is not None:
                try:
                    init_phi_field = init_phi_field.reload_data(sdir + 'restart/phi_' + re_fstr + f'-nt{nt}')
                    init_psi_field = init_psi_field.reload_data(sdir + 'restart/psi_' + re_fstr + f'-nt{nt}')
                except IOError:
                    print('phi, psi not found')
                    pass

            max_bond_E = pickle.load(open(sdir + 'restart/maxD_E_' + re_fstr + f'.pkl', 'rb'))
            max_bond_B = pickle.load(open(sdir + 'restart/maxD_B_' + re_fstr + f'.pkl', 'rb'))
            try:
                max_bond_phi = pickle.load(open(sdir + 'restart/maxD_phi_' + re_fstr + f'.pkl', 'rb'))
                max_bond_psi = pickle.load(open(sdir + 'restart/maxD_psi_' + re_fstr + f'.pkl', 'rb'))
            except IOError:
                max_bond_phi = [np.nan]
                max_bond_psi = [np.nan]
            nrg_E_ts = pickle.load(open(sdir + 'restart/nrg_E_' + re_fstr + f'.pkl', 'rb'))
            nrg_B_ts = pickle.load(open(sdir + 'restart/nrg_B_' + re_fstr + f'.pkl', 'rb'))
            ts = pickle.load(open(sdir + 'restart/ts_' + re_fstr + f'.pkl', 'rb'))
            ts = list(ts)
            print('loaded', nt, len(nrg_E_ts))

            tmp_data = None
            if other_data is not None:
                tmp_data = []
                for data_name in other_data:
                    tmp_data += [pickle.load(open(sdir + f'restart/{data_name}_' + re_fstr + '.pkl', 'rb'))]

        max_bond_E = max_bond_E[:nt + 1]
        max_bond_B = max_bond_B[:nt + 1]
        max_bond_phi = max_bond_phi[:nt + 1]
        max_bond_psi = max_bond_psi[:nt + 1]
        nrg_E_ts = nrg_E_ts[:nt + 1]
        nrg_B_ts = nrg_B_ts[:nt + 1]
        ts = ts[:nt + 1]

        if tmp_data is not None:
            for k in range(len(other_data)):
                tmp_data[k] = tmp_data[k][:nt + 1]

        print('loaded restart', nt)

    if tmp_data is not None:

        return (init_E_field, init_B_field, init_phi_field, init_psi_field,
                ts, nrg_E_ts, nrg_B_ts,
                max_bond_E, max_bond_B, max_bond_phi, max_bond_psi, *tmp_data)
    else:

        return (init_E_field, init_B_field, init_phi_field, init_psi_field,
                ts, nrg_E_ts, nrg_B_ts,
                max_bond_E, max_bond_B, max_bond_phi, max_bond_psi,)


def save_data_Maxwell(fstr, em_sys, fdir, extra_str='', add_fstr='',
                      ts=None, nrg_E_ts=None, nrg_B_ts=None,
                      max_bond_E=None, max_bond_B=None, max_bond_phi=None, max_bond_psi=None,
                      other_data: dict['str',Any]=None, only_elc=False):
    """ save data EM
    """
    f_fstr = fstr + add_fstr

    if not os.path.exists(fdir):
        os.makedirs(fdir, exist_ok=True)

    em_sys.E.save_data(fdir + 'E_' + f_fstr)
    em_sys.B.save_data(fdir + 'B_' + f_fstr)
    if em_sys.phi is not None:
        em_sys.phi.save_data(fdir + 'phi_' + f_fstr)
    if em_sys.psi is not None:
        em_sys.psi.save_data(fdir + 'psi_' + f_fstr)

    if not all(np.isnan(max_bond_E[-1])):
        pickle.dump(max_bond_E, open(fdir + 'maxD_E_' + fstr + '.pkl', 'wb'))
    if not all(np.isnan(max_bond_B[-1])):
        pickle.dump(max_bond_B, open(fdir + 'maxD_B_' + fstr + '.pkl', 'wb'))
    if max_bond_phi is not None and not np.isnan(max_bond_phi[-1]):
        pickle.dump(max_bond_phi, open(fdir + 'maxD_phi_' + fstr + '.pkl', 'wb'))
    if max_bond_psi is not None and not np.isnan(max_bond_psi[-1]):
        pickle.dump(max_bond_psi, open(fdir + 'maxD_psi_' + fstr + '.pkl', 'wb'))

    pickle.dump(nrg_E_ts, open(fdir + 'nrg_E_' + fstr + '.pkl', 'wb'))
    pickle.dump(nrg_B_ts, open(fdir + 'nrg_B_' + fstr + '.pkl', 'wb'))
    pickle.dump(ts, open(fdir + 'ts_' + fstr + '.pkl', 'wb'))

    if other_data is not None:
        for k, v in other_data.items():
            pickle.dump(v, open(fdir + str(k) + '_' + fstr + '.pkl', 'wb'))


def load_data_ED(VD_test,
                 init_fe_field: 'ScalarField', init_fi_field: 'ScalarField',
                 init_ET_field: 'Field', init_EL_field: 'Field', init_B_field: 'Field',
                 fdir: str = '', sdir: str = '',  extra_str='', save_every_nt=250,
                 load_data=True, restart=True, restart_from_T=0):
    ### get file naming string
    fstr = VD_test.get_fstr(extra_str=extra_str)
    print('filename', fstr)

    ### load data ###
    try:
        if not load_data:  raise IOError
        init_fe_field = init_fe_field.reload_data(fdir + 'fe_' + fstr)
        init_fi_field = init_fi_field.reload_data(fdir + 'fi_' + fstr)
        init_ET_field = init_ET_field.reload_data(fdir + 'ET_' + fstr)
        init_EL_field = init_EL_field.reload_data(fdir + 'EL_' + fstr)
        init_B_field = init_B_field.reload_data(fdir + 'B_' + fstr)

        max_bond_fe = pickle.load(open(fdir + 'maxD_fe_' + fstr + '.pkl', 'rb'))
        max_bond_fi = pickle.load(open(fdir + 'maxD_fi_' + fstr + '.pkl', 'rb'))
        max_bond_ET = pickle.load(open(fdir + 'maxD_ET_' + fstr + '.pkl', 'rb'))
        max_bond_EL = pickle.load(open(fdir + 'maxD_EL_' + fstr + '.pkl', 'rb'))
        max_bond_B = pickle.load(open(fdir + 'maxD_B_' + fstr + '.pkl', 'rb'))

        nrg_fe_ts = pickle.load(open(fdir + 'nrg_e_' + fstr + '.pkl', 'rb'))
        nrg_fi_ts = pickle.load(open(fdir + 'nrg_i_' + fstr + '.pkl', 'rb'))
        nrg_E_ts = pickle.load(open(fdir + 'nrg_E_' + fstr + '.pkl', 'rb'))
        nrg_B_ts = pickle.load(open(fdir + 'nrg_B_' + fstr + '.pkl', 'rb'))
        ts = pickle.load(open(fdir + 'ts_' + fstr + '.pkl', 'rb'))

        print('loaded data')

    except(IOError, EOFError):
        if restart_from_T > 0:
            restart_fstr = fstr
            init_fe_field = init_fe_field.reload_data(fdir + 'fe_' + restart_fstr)
            init_fi_field = init_fi_field.reload_data(fdir + 'fi_' + restart_fstr)
            init_ET_field = init_ET_field.reload_data(fdir + 'ET_' + restart_fstr)
            init_EL_field = init_EL_field.reload_data(fdir + 'EL_' + restart_fstr)
            init_B_field = init_B_field.reload_data(fdir + 'B_' + restart_fstr)

            max_bond_fe = pickle.load(open(fdir + 'maxD_fe_' + restart_fstr + '.pkl', 'rb'))
            max_bond_fi = pickle.load(open(fdir + 'maxD_fi_' + restart_fstr + '.pkl', 'rb'))
            max_bond_ET = pickle.load(open(fdir + 'maxD_ET_' + restart_fstr + '.pkl', 'rb'))
            max_bond_EL = pickle.load(open(fdir + 'maxD_EL_' + restart_fstr + '.pkl', 'rb'))
            max_bond_B = pickle.load(open(fdir + 'maxD_B_' + restart_fstr + '.pkl', 'rb'))

            nrg_fe_ts = pickle.load(open(fdir + 'nrg_e_' + fstr + '.pkl', 'rb'))
            nrg_fi_ts = pickle.load(open(fdir + 'nrg_i_' + fstr + '.pkl', 'rb'))
            nrg_E_ts = pickle.load(open(fdir + 'nrg_E_' + fstr + '.pkl', 'rb'))
            nrg_B_ts = pickle.load(open(fdir + 'nrg_B_' + fstr + '.pkl', 'rb'))
            ts = pickle.load(open(fdir + 'ts_' + fstr + '.pkl', 'rb'))
            print('restarting from ', restart_fstr)
            ts = list(ts)
            nt = len(ts) - 1
        else:
            if not restart:  raise ValueError

            re_fstr = fstr
            print(re_fstr)

            restart_fstrs = glob.glob(sdir + 'restart/fe_' + re_fstr + '-nt*.pkl')
            print(sdir + 'restart/')
            print('restart?', len(restart_fstrs))
            nts = []
            for f in restart_fstrs:
                nt_ind = f.find('nt')
                nt_str = f[nt_ind + 2:-4]
                nts += [int(nt_str)]
            nt = max(nts)
            print('nt', nt)
            nts.remove(nt)

            try:
                # raise NameError
                for nt_ in nts:  ## remove old restart files
                    if nt_ > nt or nt_ % save_every_nt != 0:
                        print('removing restart', nt_)
                        restart_files = glob.glob(sdir + 'restart/*' + re_fstr + f'-nt{nt_}.pkl')
                        for f in restart_files:  os.remove(f)
            except NameError:
                pass

            init_fe_field = init_fe_field.reload_data(sdir + 'restart/fe_' + re_fstr + f'-nt{nt}')
            init_fi_field = init_fi_field.reload_data(sdir + 'restart/fi_' + re_fstr + f'-nt{nt}')
            init_ET_field = init_ET_field.reload_data(sdir + 'restart/ET_' + re_fstr + f'-nt{nt}')
            init_EL_field = init_EL_field.reload_data(sdir + 'restart/EL_' + re_fstr + f'-nt{nt}')
            init_B_field = init_B_field.reload_data(sdir + 'restart/B_' + re_fstr + f'-nt{nt}')

            max_bond_fe = pickle.load(open(sdir + 'restart/maxD_fe_' + re_fstr + f'.pkl', 'rb'))
            max_bond_fi = pickle.load(open(sdir + 'restart/maxD_fi_' + re_fstr + f'.pkl', 'rb'))
            max_bond_ET = pickle.load(open(sdir + 'restart/maxD_ET_' + re_fstr + f'.pkl', 'rb'))
            max_bond_EL = pickle.load(open(sdir + 'restart/maxD_EL_' + re_fstr + f'.pkl', 'rb'))
            max_bond_B = pickle.load(open(sdir + 'restart/maxD_B_' + re_fstr + f'.pkl', 'rb'))

            nrg_fe_ts = pickle.load(open(sdir + 'restart/nrg_e_' + re_fstr + f'.pkl', 'rb'))
            nrg_fi_ts = pickle.load(open(sdir + 'restart/nrg_i_' + re_fstr + f'.pkl', 'rb'))
            nrg_E_ts = pickle.load(open(sdir + 'restart/nrg_E_' + re_fstr + f'.pkl', 'rb'))
            nrg_B_ts = pickle.load(open(sdir + 'restart/nrg_B_' + re_fstr + f'.pkl', 'rb'))
            ts = pickle.load(open(sdir + 'restart/ts_' + re_fstr + f'.pkl', 'rb'))
            ts = list(ts)
            print('loaded', nt, len(nrg_fe_ts))

        max_bond_fe = max_bond_fe[:nt + 1]
        max_bond_fi = max_bond_fi[:nt + 1]
        max_bond_ET = max_bond_ET[:nt + 1]
        max_bond_EL = max_bond_EL[:nt + 1]
        max_bond_B = max_bond_B[:nt + 1]
        nrg_fe_ts = nrg_fe_ts[:nt + 1]
        nrg_fi_ts = nrg_fi_ts[:nt + 1]
        nrg_E_ts = nrg_E_ts[:nt + 1]
        nrg_B_ts = nrg_B_ts[:nt + 1]
        ts = ts[:nt + 1]

        print('loaded restart', nt)

    return (init_fe_field, init_fi_field, init_ET_field, init_EL_field, init_B_field,
            ts, nrg_fe_ts, nrg_fi_ts, nrg_E_ts, nrg_B_ts,
            max_bond_fe, max_bond_fi, max_bond_ET, max_bond_EL, max_bond_B,)


def save_data_ED(VP_test, vd_sys, fdir, extra_str='', add_fstr='',
                 ts=None, nrg_fe_ts=None, nrg_fi_ts=None, nrg_E_ts=None, nrg_B_ts=None,
                 max_bond_fe=None, max_bond_fi=None, max_bond_ET=None, max_bond_EL=None, max_bond_B=None,
                 ):
    """ save data EM
    """
    fstr = VP_test.get_fstr(extra_str=extra_str)
    f_fstr = fstr + add_fstr

    vd_sys.fe.save_data(fdir + 'fe_' + f_fstr)
    vd_sys.fi.save_data(fdir + 'fi_' + f_fstr)
    vd_sys.ET.save_data(fdir + 'ET_' + f_fstr)
    vd_sys.EL.save_data(fdir + 'EL_' + f_fstr)
    vd_sys.B.save_data(fdir + 'B_' + f_fstr)
    if not np.isnan(max_bond_fe[-1]):
        pickle.dump(max_bond_fe, open(fdir + 'maxD_fe_' + fstr + '.pkl', 'wb'))
    if not np.isnan(max_bond_fi[-1]):
        pickle.dump(max_bond_fi, open(fdir + 'maxD_fi_' + fstr + '.pkl', 'wb'))
    if not all(np.isnan(max_bond_ET[-1])):
        pickle.dump(max_bond_ET, open(fdir + 'maxD_ET_' + fstr + '.pkl', 'wb'))
    if not all(np.isnan(max_bond_EL[-1])):
        pickle.dump(max_bond_EL, open(fdir + 'maxD_EL_' + fstr + '.pkl', 'wb'))
    if not all(np.isnan(max_bond_B[-1])):
        pickle.dump(max_bond_B, open(fdir + 'maxD_B_' + fstr + '.pkl', 'wb'))
    pickle.dump(nrg_fe_ts, open(fdir + 'nrg_e_' + fstr + '.pkl', 'wb'))
    pickle.dump(nrg_fi_ts, open(fdir + 'nrg_i_' + fstr + '.pkl', 'wb'))
    pickle.dump(nrg_E_ts, open(fdir + 'nrg_E_' + fstr + '.pkl', 'wb'))
    pickle.dump(nrg_B_ts, open(fdir + 'nrg_B_' + fstr + '.pkl', 'wb'))
    pickle.dump(ts, open(fdir + 'ts_' + fstr + '.pkl', 'wb'))


