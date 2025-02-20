import numpy as np
import warnings
import pickle
import shutil
import sys

from get_PFM_info import get_PFM_info

def copy_mv_nc_file(nc_type,lvl):
    # this copies an atm.nc or river.nc file to the archive location
    # nc_type can be ['atm','river']
    # lvl can be ['lv1','lv2','lv3','lv4']

    lvl_upper = lvl.upper()

    PFM = get_PFM_info()
    fcdate = PFM['fetch_time'].strftime("%Y%m%d%H")
    dir_out = '/dataSIO/PFM_Simulations/Archive2.5/Forcing/'
    if nc_type == 'atm':
        fn_in_full = PFM[lvl+'_forc_dir'] + '/' + PFM[lvl+'_atm_file']
        fn_out = 'atm_' + PFM['atm_model'] + '_' + lvl_upper + '_' + fcdate + '.nc'
    if nc_type == 'river':
        fn_in_full = PFM[lvl+'_forc_dir'] + '/' + PFM[lvl+'_river_file']
        fn_out = 'river_' + lvl_upper + '_' + fcdate + '.nc'

    fn_out_full = dir_out + fn_out
    shutil.move(fn_in_full, fn_out_full)


def display_timing_info():
    PFM = get_PFM_info()
    fn_timing = PFM['lv1_run_dir'] + '/LV1_timing_info.pkl'
    with open(fn_timing,'rb') as fout:
        T1 = pickle.load(fout)
    fn_timing = PFM['lv2_run_dir'] + '/LV2_timing_info.pkl'
    with open(fn_timing,'rb') as fout:
        T2 = pickle.load(fout)
    fn_timing = PFM['lv3_run_dir'] + '/LV3_timing_info.pkl'
    with open(fn_timing,'rb') as fout:
        T3 = pickle.load(fout)
    fn_timing = PFM['lv4_run_dir'] + '/LV4_timing_info.pkl'
    with open(fn_timing,'rb') as fout:
        T4 = pickle.load(fout)

    GI = PFM['gridinfo']
    SS = PFM['stretching']

    #print(T1['roms'])

    tp1 = np.sum(T1['process'][:])
    tp1 = np.round( tp1.total_seconds()/60,decimals=2 )
    tatm1 = np.sum(T1['atm'][:])
    tatm1 = np.round( tatm1.total_seconds()/60,decimals=2 )
    tatm2 = np.sum(T2['atm'][:])
    tatm2 = np.round( tatm2.total_seconds()/60,decimals=2 )
    tatm3 = np.sum(T3['atm'][:])
    tatm3 = np.round( tatm3.total_seconds()/60,decimals=2 )
    tatm4 = np.sum(T4['atm'][:])
    tatm4 = np.round( tatm4.total_seconds()/60,decimals=2 )
    
    to1 = np.sum(T1['ic'][:])
    to1 = np.round( to1.total_seconds()/60,decimals=2 )
    to2 = np.sum(T2['ic'][:])
    to2 = np.round( to2.total_seconds()/60,decimals=2 )
    to3 = np.sum(T3['ic'][:])
    to3 = np.round( to3.total_seconds()/60,decimals=2 )
    to4 = np.sum(T4['ic'][:])
    to4 = np.round( to4.total_seconds()/60,decimals=2 )
    
    tb1 = np.sum(T1['bc'][:])
    tb1 = np.round( tb1.total_seconds()/60,decimals=2 )
    tb2 = np.sum(T2['bc'][:])
    tb2 = np.round( tb2.total_seconds()/60,decimals=2 )
    tb3 = np.sum(T3['bc'][:])
    tb3 = np.round( tb3.total_seconds()/60,decimals=2 )
    tb4 = np.sum(T4['bc'][:])
    tb4 = np.round( tb4.total_seconds()/60,decimals=2 )

    tp1 = np.sum(T1['plotting'][:])
    tp1 = np.round( tp1.total_seconds()/60,decimals=2 )
    tp2 = np.sum(T2['plotting'][:])
    tp2 = np.round( tp2.total_seconds()/60,decimals=2 )
    tp3 = np.sum(T3['plotting'][:])
    tp3 = np.round( tp3.total_seconds()/60,decimals=2 )
    tp4 = np.sum(T4['plotting'][:])
    tp4 = np.round( tp4.total_seconds()/60,decimals=2 )

    tr1 = np.sum(T1['roms'][:])
    #print(tr1)
    #print(str(tr1))
    tr1 = np.round( tr1.total_seconds()/60,decimals=2 )
    #print(tr1)
    tr2 = np.sum(T2['roms'][:])
    tr2 = np.round( tr2.total_seconds()/60,decimals=2 )
    tr3 = np.sum(T3['roms'][:])
    tr3 = np.round( tr3.total_seconds()/60,decimals=2 )
    tr4 = np.sum(T4['roms'][:])
    tr4 = np.round( tr4.total_seconds()/60,decimals=2 )


    t_atm = np.round(T1['download_atm'][0].total_seconds()/60,decimals=2)
    t_ocn = np.round(T1['download_ocn'][0].total_seconds()/60,decimals=2)

    tot1 = t_atm + t_ocn + tp1 + tatm1 + to1 + tb1 + tr1 + tp1
    tot2 = tatm2 + to2 + tb2 + tr2 + tp2
    tot3 = tatm3 + to3 + tb3 + tr3 + tp3
    tot4 = tatm4 + to4 + tb4 + tr4 + tp4
    if "swan" in T4:
        #print( T4['swan'][0].total_seconds() )
        tsw = np.round( T4['swan'][0].total_seconds()/60,decimals=2 )
        tot4 = tot4 + tsw

    tot_tot = tot1+tot2+tot3+tot4

    tot1 = np.round(tot1,decimals=2)
    tot2 = np.round(tot2,decimals=2)
    tot3 = np.round(tot3,decimals=2)
    tot4 = np.round(tot4,decimals=2)
    tot_tot = np.round(tot_tot,decimals=2)

    ttt = PFM['tinfo']
    stp1 = int( PFM['forecast_days'] * 24 * 3600 / ttt['L1','dtsec'] )
    stp2 = int( PFM['forecast_days'] * 24 * 3600 / ttt['L2','dtsec'] )
    stp3 = int( PFM['forecast_days'] * 24 * 3600 / ttt['L3','dtsec'] )
    stp4 = int( PFM['forecast_days'] * 24 * 3600 / ttt['L4','dtsec'] )

    #PFM['lv1_use_restart'] = 0
    if PFM['lv1_use_restart'] == 1:
        ocn_ic = 'restart file'
    else:
        ocn_ic = PFM['ocn_model']

    print('\n')
    print(f'----------------------------------------------------------------------')
    print(f'{''}{'PFM general information' : ^70}{''}')
    print(f'----------------------------------------------------------------------')
    print(PFM['start_time'].strftime("%Y-%m-%d %H:%M"), '    :    simulation started [local time]')
    print(PFM['fetch_time'].strftime("%Y-%m-%d %H:%M"), '    :    forecast start time [UTC]')
    print(PFM['fore_end_time'].strftime("%Y-%m-%d %H:%M"), '    :    forecast end time [UTC]')
    print(PFM['forecast_days'], '                 :    forecast length [days]')
    print(f'{PFM['lv4_model'] : <21}{':' : <5}{'LV4 ocean model' : <30}')
    print(f'{PFM['ocn_model'] : <21}{':' : <5}{'ocean boundary conditions' : <30}')
    print(f'{T1['hycom_t0'][0:4] + '-' + T1['hycom_t0'][4:6] + '-' + T1['hycom_t0'][6::] + ' 12:00'   : <21}{':' : <5}{'hycom forecast start time [UTC]' : <30}')
#    print(f'{'20241022' + ' 12:00'   : <21}{':' : <5}{'hycom forecast time' : <30}')
    print(f'{ocn_ic : <21}{':' : <5}{'ocean initial condition' : <30}')
    print(f'{PFM['atm_model'] : <21}{':' : <5}{'atmospheric forcing model' : <30}')
    print(f'----------------------------------------------------------------------')
    print(f'             |             |             |             |             |')
    print(f'             |     LV1     |     LV2     |     LV3     |     LV4     |')
    print(f'             |             |             |             |             |')
    print(f'----------------------------------------------------------------------')
    print(f'{''}{'grid points' : ^70}{''}')
    print(f'{'nx =' : >8}{'|' : >6}{GI['L1','Lm'] : >8}{'|' : >6} {GI['L2','Lm'] : >7}{'|' : >6}{GI['L3','Lm'] : >8}{'|' : >6}{GI['L4','Lm'] : >8}{'|' : >6}')
    print(f'{'ny =' : >8}{'|' : >6}{GI['L1','Mm'] : >8}{'|' : >6} {GI['L2','Mm'] : >7}{'|' : >6}{GI['L3','Mm'] : >8}{'|' : >6}{GI['L4','Mm'] : >8}{'|' : >6}')
    print(f'{'nz =' : >8}{'|' : >6}{SS['L1','Nz'] : >8}{'|' : >6} {SS['L2','Nz'] : >7}{'|' : >6}{SS['L3','Nz'] : >8}{'|' : >6}{SS['L4','Nz'] : >8}{'|' : >6}')
    print(f'----------------------------------------------------------------------')
    print(f'{''}{'time steps' : ^70}{''}')
    print(f'{'' : <7}{'|' : >7}{stp1 : >8}{'|' : >6} {stp2 : >7}{'|' : >6}{stp3 : >8}{'|' : >6}{stp4 : >8}{'|' : >6}')
    print(f'----------------------------------------------------------------------')
    print(f'{''}{'parallelization' : ^70}{''}')
    print(f'{'nodes' : >11}{'|' : >3}{GI['L1','nnodes'] : >8}{'|' : >6} {GI['L2','nnodes'] : >7}{'|' : >6}{GI['L3','nnodes'] : >8}{'|' : >6}{GI['L4','nnodes'] : >8}{'|' : >6}')
    print(f'{'Ntilei' : >11}{'|' : >3}{GI['L1','ntilei'] : >8}{'|' : >6} {GI['L2','ntilei'] : >7}{'|' : >6}{GI['L3','ntilei'] : >8}{'|' : >6}{GI['L4','ntilei'] : >8}{'|' : >6}')
    print(f'{'Ntilej' : >11}{'|' : >3}{GI['L1','ntilej'] : >8}{'|' : >6} {GI['L2','ntilej'] : >7}{'|' : >6}{GI['L3','ntilej'] : >8}{'|' : >6}{GI['L4','ntilej'] : >8}{'|' : >6}')
    if "swan" in T4:
        print(f'{'  swan CPUs' : >11}{'|' : >3}{'' : >8}{'|' : >6} {'' : >7}{'|' : >6}{'' : >8}{'|' : >6}{GI['L4','np_swan'] : >8}{'|' : >6}')
    print(f'----------------------------------------------------------------------')
    print(f'----------------------------------------------------------------------')
    print(f'{''}{'timing information for PFM' : ^70}{''}')
    print(f'{''}{'[in minutes]' : ^70}{''}')
    print(f'----------------------------------------------------------------------')
    print(f'----------------------------------------------------------------------')
    print(f'{'downloading' : >8}{'|' : >3}{'': >8}{'|' : >6} {'' : >7}{'|' : >6}{'' : >8}{'|' : >6}{'' : >8}{'|' : >6}')
    print(f'{'atm' : >7}{'|' : >7}{t_atm : >8}{'|' : >6} {'' : >7}{'|' : >6}{'' : >8}{'|' : >6}{'' : >8}{'|' : >6}')
    print(f'{'ocn' : >7}{'|' : >7}{t_ocn : >8}{'|' : >6} {'' : >7}{'|' : >6}{'' : >8}{'|' : >6}{'' : >8}{'|' : >6}')
    print(f'----------------------------------------------------------------------')
    print(f'{'processing' : >7}{'|' : >4}{tp1 : >8}{'|' : >6} {'' : >7}{'|' : >6}{'' : >8}{'|' : >6}{'' : >8}{'|' : >6}')
    print(f'----------------------------------------------------------------------')
    print(f'{'atm' : <7}{'|' : >7}{tatm1 : >8}{'|' : >6} {tatm2 : >7}{'|' : >6}{tatm3 : >8}{'|' : >6}{tatm4 : >8}{'|' : >6}')
    print(f'----------------------------------------------------------------------')
    print(f'{'ocn ic' : <7}{'|' : >7}{to1 : >8}{'|' : >6} {to2 : >7}{'|' : >6}{to3 : >8}{'|' : >6}{to4 : >8}{'|' : >6}')
    print(f'----------------------------------------------------------------------')
    print(f'{'ocn bc' : <7}{'|' : >7}{tb1 : >8}{'|' : >6} {tb2 : >7}{'|' : >6}{tb3 : >8}{'|' : >6}{tb4 : >8}{'|' : >6}')
    print(f'----------------------------------------------------------------------')
    if "swan" in T4:
        print(f'{'swan files' : <7}{'|' : >4}{'' : >8}{'|' : >6} {'' : >7}{'|' : >6}{'' : >8}{'|' : >6}{tsw : >8}{'|' : >6}')
        print(f'----------------------------------------------------------------------')
    print(f'{'plotting' : <7}{'|' : >6}{tp1 : >8}{'|' : >6} {tp2 : >7}{'|' : >6}{tp3 : >8}{'|' : >6}{tp4 : >8}{'|' : >6}')
    print(f'----------------------------------------------------------------------')
    print(f'{'ROMS' : <7}{'|' : >7}{tr1 : >8}{'|' : >6} {tr2 : >7}{'|' : >6}{tr3 : >8}{'|' : >6}{tr4 : >8}{'|' : >6}')
    print(f'----------------------------------------------------------------------')
    print(f'{'totals' : <7}{'|' : >7}{tot1 : >8}{'|' : >6} {tot2 : >7}{'|' : >6}{tot3 : >8}{'|' : >6}{tot4 : >8}{'|' : >6}')
    print(f'----------------------------------------------------------------------')
    print(f'{'TOTAL [min]' : <11}{'|' : >3}{tot_tot : ^53}{'|' : >3}')
    print(f'----------------------------------------------------------------------')
    print('\n')


#print(f'max {vnm:6} = {mxx:6.3f} {ulist2[vnm]:5}      at  ( it, ilat, ilon)     =  ({ind_mx[0]:3},{ind_mx[1]:4},{ind_mx[2]:4})')



   
class s_coordinate(object):
    def __init__(self, h, theta_b, theta_s, Tcline, N, hraw=None, zeta=None):
        self.hraw = hraw
        self.h = np.asarray(h)
        self.hmin = self.h.min()
        self.theta_b = theta_b
        self.theta_s = theta_s
        self.Tcline = Tcline
        self.N = int(N)
        self.Np = self.N + 1
        self.hc = min(self.hmin, self.Tcline)
        self.Vtrans = 1

        if self.Tcline > self.hmin:
            warnings.warn(
                'Vertical transformation parameters are not defined correctly: \n Tcline = %d and hmin = %d. \n You need to make sure that Tcline <= hmin when using transformation 1.' % (self.Tcline, self.hmin))

        self.c1 = 1.0
        self.c2 = 2.0
        self.p5 = 0.5

        if zeta is None:
            self.zeta = np.zeros(h.shape)
        else:
            self.zeta = zeta

        self._get_s_rho()
        self._get_s_w()
        self._get_Cs_r()
        self._get_Cs_w()

        self.z_r = z_r(self.h, self.hc, self.N, self.s_rho, self.Cs_r, self.zeta, self.Vtrans)
        self.z_w = z_w(self.h, self.hc, self.Np, self.s_w, self.Cs_w, self.zeta, self.Vtrans)

    def _get_s_rho(self):
        lev = np.arange(1, self.N + 1, 1)
        ds = 1.0 / self.N
        self.s_rho = -self.c1 + (lev - self.p5) * ds

    def _get_s_w(self):
        lev = np.arange(0, self.Np, 1)
        ds = 1.0 / (self.Np - 1)
        self.s_w = -self.c1 + lev * ds

    def _get_Cs_r(self):
        if self.theta_s >= 0:
            Ptheta = np.sinh(self.theta_s * self.s_rho) / np.sinh(self.theta_s)
            Rtheta = np.tanh(self.theta_s * (self.s_rho + self.p5)) / \
                (self.c2 * np.tanh(self.p5 * self.theta_s)) - self.p5
            self.Cs_r = (self.c1 - self.theta_b) * Ptheta + self.theta_b * Rtheta
        else:
            self.Cs_r = self.s_rho

    def _get_Cs_w(self):
        if self.theta_s >= 0:
            Ptheta = np.sinh(self.theta_s * self.s_w) / np.sinh(self.theta_s)
            Rtheta = np.tanh(self.theta_s * (self.s_w + self.p5)) / \
                (self.c2 * np.tanh(self.p5 * self.theta_s)) - self.p5
            self.Cs_w = (self.c1 - self.theta_b) * Ptheta + self.theta_b * Rtheta
        else:
            self.Cs_w = self.s_w

class s_coordinate_4(s_coordinate):
    def __init__(self, h, theta_b, theta_s, Tcline, N, hraw=None, zeta=None):
        self.hraw = hraw
        self.h = np.asarray(h)
        self.hmin = h.min()
        self.theta_b = theta_b
        self.theta_s = theta_s
        self.Tcline = Tcline
        self.N = int(N)
        self.Np = self.N + 1
        self.hc = self.Tcline
        self.Vtrans = 4
        self.c1 = 1.0
        self.c2 = 2.0
        self.p5 = 0.5

        if zeta is None:
            self.zeta = np.zeros(h.shape)
        else:
            self.zeta = zeta

        self._get_s_rho()
        self._get_s_w()
        self._get_Cs_r()
        self._get_Cs_w()

        self.z_r = z_r(self.h, self.hc, self.N, self.s_rho, self.Cs_r, self.zeta, self.Vtrans)
        self.z_w = z_w(self.h, self.hc, self.Np, self.s_w, self.Cs_w, self.zeta, self.Vtrans)

    def _get_Cs_r(self):
        if self.theta_s > 0:
            Csur = (self.c1 - np.cosh(self.theta_s * self.s_rho)) / \
                (np.cosh(self.theta_s) - self.c1)
        else:
            Csur = -self.s_rho**2
        if self.theta_b > 0:
            Cbot = (np.exp(self.theta_b * Csur) - self.c1) / \
                (self.c1 - np.exp(-self.theta_b))
            self.Cs_r = Cbot
        else:
            self.Cs_r = Csur

    def _get_Cs_w(self):
        if self.theta_s > 0:
            Csur = (self.c1 - np.cosh(self.theta_s * self.s_w)) / \
                (np.cosh(self.theta_s) - self.c1)
        else:
            Csur = -self.s_w**2
        if self.theta_b > 0:
            Cbot = (np.exp(self.theta_b * Csur) - self.c1) / \
                (self.c1 - np.exp(-self.theta_b))
            self.Cs_w = Cbot
        else:
            self.Cs_w = Csur

class z_r(object):
    def __init__(self, h, hc, N, s_rho, Cs_r, zeta, Vtrans):
        self.h = h
        self.hc = hc
        self.N = N
        self.s_rho = s_rho
        self.Cs_r = Cs_r
        self.zeta = zeta
        self.Vtrans = Vtrans

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(self.zeta.shape) > len(self.h.shape):
            zeta = self.zeta[key[0]]
            res_index = (slice(None),) + key[1:]
        elif len(self.zeta.shape) > len(self.h.shape):
            zeta = self.zeta[key]
            res_index = slice(None)
        else:
            zeta = self.zeta
            res_index = key

        if self.h.ndim == zeta.ndim:
            zeta = zeta[np.newaxis, :]

        ti = zeta.shape[0]
        z_r = np.empty((ti, self.N) + self.h.shape, 'd')
        if self.Vtrans == 1:
            for n in range(ti):
                for k in range(self.N):
                    z0 = self.hc * self.s_rho[k] + (self.h - self.hc) * self.Cs_r[k]
                    z_r[n, k, :] = z0 + zeta[n, :] * (1.0 + z0 / self.h)
        elif self.Vtrans in [2, 4]:
            for n in range(ti):
                for k in range(self.N):
                    z0 = (self.hc * self.s_rho[k] + self.h * self.Cs_r[k]) / \
                        (self.hc + self.h)
                    z_r[n, k, :] = zeta[n, :] + (zeta[n, :] + self.h) * z0

        return np.squeeze(z_r[res_index])

class z_w(object):
    def __init__(self, h, hc, Np, s_w, Cs_w, zeta, Vtrans):
        self.h = h
        self.hc = hc
        self.Np = Np
        self.s_w = s_w
        self.Cs_w = Cs_w
        self.zeta = zeta
        self.Vtrans = Vtrans

    def __getitem__(self, key):
        if isinstance(key, tuple) and len(self.zeta.shape) > len(self.h.shape):
            zeta = self.zeta[key[0]]
            res_index = (slice(None),) + key[1:]
        elif len(self.zeta.shape) > len(self.h.shape):
            zeta = self.zeta[key]
            res_index = slice(None)
        else:
            zeta = self.zeta
            res_index = key

        if self.h.ndim == zeta.ndim:
            zeta = zeta[np.newaxis, :]

        ti = zeta.shape[0]
        z_w = np.empty((ti, self.Np) + self.h.shape, 'd')
        if self.Vtrans == 1:
            for n in range(ti):
                for k in range(self.Np):
                    z0 = self.hc * self.s_w[k] + (self.h - self.hc) * self.Cs_w[k]
                    z_w[n, k, :] = z0 + zeta[n, :] * (1.0 + z0 / self.h)
        elif self.Vtrans in [2, 4]:
            for n in range(ti):
                for k in range(self.Np):
                    z0 = (self.hc * self.s_w[k] + self.h * self.Cs_w[k]) / \
                        (self.hc + self.h)
                    z_w[n, k, :] = zeta[n, :] + (zeta[n, :] + self.h) * z0

        return np.squeeze(z_w[res_index])
    
if __name__ == "__main__":
    args = sys.argv
    # args[0] = current file
    # args[1] = function name
    # args[2:] = function args : (*unpacked)
    globals()[args[1]](*args[2:])

