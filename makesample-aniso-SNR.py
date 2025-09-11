#!/usr/bin/python
from __future__ import print_function
import os, sys, time, math
import numpy as np
try:
   import cPickle as pickle
except:
   import pickle

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as color

import random

import healpy as hp

NSIDE = 64
m = np.arange(hp.nside2npix(NSIDE))
totnpix=len(m)

lo = np.zeros(hp.nside2npix(NSIDE))
la = np.zeros(hp.nside2npix(NSIDE))
cosdistfromvela=np.zeros(hp.nside2npix(NSIDE))
#poscosdistfromvela=np.zeros(hp.nside2npix(NSIDE))
#negcosdistfromvela=np.zeros(hp.nside2npix(NSIDE))

###############################################################################
#
#
#                  Definitions and Functions
#
#
###############################################################################

nyear=7.34              #take signal with this prefix !!!    
                        #(no longer multiplied with signal => signal contains number of years)
animapnyear=3.05
                        
Elowbound=99.0

dragonoutpath='/home/motz/dragon_output/'
dragonpath='/ssdhome/motz/dragon-3.0.1/'
prop="LD"

#cond="T0d0R7000d0C20d0"
#cond="T0d0R5000d0C100d0"
cond="T10000d0C100d0"
expofact=4.8

anisodatafolder="/home/motz/nfs2/anisodata/"

dirs={}
dirs["Vela"]=[-96.1,-3.3]
dirs["Geminga"]=[-164.88,4.32] 
dirs["Mogem"]=[-158.9,8.3]
dirs["CygLo"]=[74.0,-8.5]

latcut=False
dragonSNR=[True,["Vela"]] #,"CygLo","Mogem"]] 

dragonSNRfileLD="RSNR-%s-%s-D0d66-d0d5B500d0H0d2S0d05-DE1d8N0d15R4d5B2d0-g2d84-i0d0-e1d1-%s-NS2000-L3-NF.txt"
dragonSNRfileMD="RSNR-%s-%s-D1d32-d0d5B500d0H0d2S0d05-DE3d5N0d15R4d5B2d0-g2d84-i0d0-e1d1-%s-NS2000-NF.txt"
dragonSNRfileHD="RSNR-%s-%s-D1d78-d0d5B500d0H0d2S0d05-DE5d3N0d15R4d5B2d0-g2d84-i0d0-e1d1-%s-NS2000-L9-NF.txt"
dragonSNRfileXD="RSNR-%s-%s-D0d16-d0d5B1000d0H0d01S0d3-DE0d6N0d15R3d0B2d0-g2d835-i0d0-e1d1-R18d0-%s-NS2000-NF.txt"
dragonSNRfileZDLB="ESNR-%s-%s-D3d066400-d0d514100B913d5324H0d0018S0d413LB8d4373L0d2112S0d0529-DE4d5N0d15R10d7B2d0-g2d8584-i0d0-e1d1-R14d74-%s-NS2000-NF.txt"


delt=0.5
highdelt=0.2
deltbreak=500.0
deltsoft=0.05
lowdelt=False
lowdeltbreak=False
lowdeltsoft=False
dire=False
DnormE=4.0


if prop=="HD":
    signalfile ="./"+str(nyear)+"yr-signal-DSNRpuls-All%s%s-RCALYA23AERAMS-NP-EC1000-SY10-binavg-FPS5d0B500d0H0d20S0d05-DC1d78-LESB0d05-SF-SRT.dat"
    dragonSNRfile="aniso-"+dragonSNRfileHD       
    Dnormbase=1.78
    fixPS=0.5
    fixDC=Dnormbase
    diffnormrad=4.5
    Dnorm=Dnormbase*math.exp(6.3/diffnormrad)*1e28
elif prop=="LD":    
    #signalfile ="./"+str(nyear)+"yr-signal-DSNRpuls-All%s%s-XCALEAMS-NP-EC1000-SX10P0d500E0d500-binavg-FPS6d2B350d0H0d33S0d15-DC1d1-LESB0d05-FNUI-SF-RT60000.dat"
    signalfile ="./"+str(nyear)+"yr-signal-DSNRpuls-All%s%s-RCALYA23AERAMS-NP-EC1000-SY10-binavg-FPS5d0B500d0H0d20S0d05-DC0d66-LESB0d05-SF-SRT.dat"
    dragonSNRfile="aniso-"+dragonSNRfileLD
    Dnormbase=0.66
    fixPS=0.5
    fixDC=Dnormbase 
    diffnormrad=4.5
    Dnorm=Dnormbase*math.exp(6.3/diffnormrad)*1e28
elif prop=="XD":    
    signalfile ="./"+str(nyear)+"yr-signal-DSNRpuls-All%s%s-RCALYA23AERAMS-NP-EC1000-SY10-binavg-FPS5d0B1000d0H0d01S0d3-DC0d16-LESB0d05-SF-SRT.dat"
    dragonSNRfile="aniso-"+dragonSNRfileXD
    Dnormbase=0.16e28
    Dnorm=Dnormbase*math.exp(6.3/3.0)
    highdelt=0.01
    deltbreak=1000.0
    deltsoft=0.3
elif prop=="ZDLB":
    signalfile ="./"+str(nyear)+"yr-signal-DSNRpuls-All%s%s-XCALYA23AERAMS-NP-EC1000-SY10-binavg-FPS5d1B913d5324H0d0018S0d4130LB8d4373L0d2112S0d0529-DC3d0664-LESB0d05-FNUI-SF-SRT.dat"
    dragonSNRfile="aniso-"+dragonSNRfileZDLB
    gp={}
    gp["diffexp"]=0.5141
    gp["sindex"]=2.8584
    gp["diffnorm"]=3.0664
    gp["DN"]=0.15
    gp["DB"]=2.0
    gp["DE"]=4.50
    gp["DR"]=10.70
    gp["alvel"]=14.7400
    gp["diffL"]=6
    gp["nuccut"]=30.1170
    gp["highexp"]=0.0018
    gp["diffbreak"]=913.5324
    gp["diffbreaksoft"]=0.4130
    gp["lowexp"]=0.2112
    gp["lowdiffbreak"]=8.4373
    gp["lowdiffbreaksoft"]=0.0529
    gp["lowindx"]=2.0300
    gp["lowbreak"]=13.4503
    gp["lowsoft"]=0.2285

    delt=gp["diffexp"]   
    Dnormbase=gp["diffnorm"]
    Dnorm=Dnormbase*math.exp(6.3/gp["DR"])*1e28
    fixDC=Dnorm
    
    dire=gp["alvel"]
    highdelt=gp["highexp"]
    deltbreak=gp["diffbreak"]
    deltsoft=gp["diffbreaksoft"]
    lowdelt=gp["lowexp"]
    lowdeltbreak=gp["lowdiffbreak"]
    lowdeltsoft=gp["lowdiffbreaksoft"]
    
elif prop=="old":
    signalfile ="./"+str(nyear)+"-signal-DSNRpuls-AllVela%s-XCALEAMS-NP-EC2000-SX10P0d500E0d500-binavg-FPS6d0B300d0H0d33S0d2-DC1d3-LESB0d05-FNUI-SF-RT60000-SNN10.dat"
    dragonSNRfile="aniso-ESNR-%s-%s-D1d3-d0d6B300d0H0d33S0d2-g2d92-i0d0-e1d1-%s-NS2000-L3-NF.txt"
    Dnorm=1.3e28
    DnormE=4.0
    delt=0.6
    highdelt=0.33
    deltbreak=300.0
    deltsoft=0.2


#"3.05yr-signal-puls-All-XCALEAMS-NP-EC10000-SX10P0d500E0d500-binavg-FPS6d0B300d0H0d33S0d2-DC1d3-LESB0d05-FNUI-SF-RT60000.dat"
exmapfolder="../exposuremap/PY3QK10-46/"
extype="abs"

timeinsec=365.25*24*3600     # 1 year
kpcincm=3.086e+21
lsincmps = 299792458*100.0

globeffarea=1040.0
globeff=0.80
suncoord=(8.3,0,0)    

oldexpo=globeffarea*globeff*nyear*timeinsec

def unpicklesignals(sfn):
    sfile=open(sfn,'rb')   
    loadthing=pickle.load(sfile)
    sfile.close
    return loadthing

def unpicklefluxes(ffn):
    sfile=open(ffn,'rb')   
    loadthing=pickle.load(sfile)
    sfile.close
    return loadthing

def preparemakesample(signalfit,aniflx,geo,minE=Elowbound):
    totexpevents=[]
    poscosveladistevt=[]
    negcosveladistevt=[]
    fitflux=converttoflux(signalfit[1],oldexpo)
    seff=signalfit[0][15]
    totflx=[]
    es=sorted(signalfit[1].keys())
    emap=np.full(len(es),-1,dtype=int)
    ai=-1
    for e in aniflx[0]:
        ai=ai+1
        f=0.0
        fi=-1
        for he in es:
            fi=fi+1
            if emap[fi]<0:
                emap[fi]=ai
            if he>e:
                le = es[es.index(he)-1]        
                hpf=fitflux[he][0]
                lpf=fitflux[le][0]
                sl=(e-le)/(he-le)
                f=pow(lpf,1.0-sl)*pow(hpf,sl)
                break
        totflx.append(f)
    exmaps,thresholds,totexps=readexmaps(extype,geo,exmapfolder)
    anis,longitudes,latitudes=calcaniso(aniflx)    
    hpcoo={}
    for pix in m:
        clis,clad=getclosestpoints(lo[pix],la[pix],longitudes,latitudes,4)
        hpcoo[pix]=(lo[pix],la[pix],clis,clad)
    energylist=aniflx[0]    
    flx=[]
    bkgflx=[]
    animaps=[]
    snf=aniflx[1][suncoord]
    for e,sf,tf in zip(aniflx[0],snf,totflx):
        flx.append(sf*seff/10000.0)    
        bkgflx.append(max(tf-sf*seff/10000.0,0.0))
        effanis=[]
        for ani in anis:
                
            effanis.append((bkgflx[-1]+flx[-1]*(ani[len(flx)-1]+1.0))/(bkgflx[-1]+flx[-1]))
        #print("average of effanis : ",sum(effanis)/float(len(effanis)))
        animap=np.zeros(totnpix)
        for pix in m:
            clis=hpcoo[pix][2]
            clad=hpcoo[pix][3]    
            animap[pix]=interpolevent(effanis,clis,clad)
        #print("average of anisotropy map for energy ",e," is ",np.average(animap)) 
        animaps.append(animap)    
    #print("start, confirm same length of flux lists: ", len(aniflx[0]),len(snf),len(totflx))
    exanimaps=[]
    sigeffs=[]
    anorms=[]
    reales=[]
    for i in range(1,len(es)-1):
        e=es[i]
        if not abs((e-10**((i)*0.001))/e)<1e-6:
            print("energy binning wrong in signal. (e=%f should be: %f )exit"%(e, 10**((i)*0.001))) 
            sys.exit(1)
        ei_min=10**((i-0.5)*0.001)
        ei_max=10**((i+0.5)*0.001)
        if ei_min < minE:
            continue
        for j in range(len(thresholds)):
            if e>thresholds[j]:
                continue
            elif totexps[j]>0.0:            
                nowexmap=exmaps[j]*(1.0/totexps[j])
                expocorr=float(expofact)*4*math.pi*totexps[j]/totnpix/oldexpo
                break
            else:
                #print("no exposure map for this energy (%f), using flat map, setting exposure to zero"%e)
                nowexmap=np.ones(totnpix)*(1.0/float(totnpix))
                expocorr=0.0
                break
        
        if expocorr==0.0:
            continue
        
        nb=signalfit[1][es[i]]*expocorr        
        aindx=int(emap[i])
        exanimaps.append(np.multiply(nowexmap,animaps[aindx]))
        anorms.append(np.sum(exanimaps[-1]))
        sigeffs.append(nb*anorms[-1])
        reales.append((e,ei_min,ei_max))
    return reales,sigeffs,exanimaps,anorms

def makesample(es,sigeffs,exanimaps,anorms,extseed):    
    random.seed(extseed)
    np.random.seed(extseed)
    totalevents=0
    nevent=0
    events=[]
    #printmap=True
    reales=[]
    
    for i in range(1,len(es)-1):
        e=es[i][0]
        sigeff=sigeffs[i]
        exanimap=exanimaps[i]
        anorm=anorms[i]
        
        if sigeff >100:
            nevent=int(round(random.gauss(sigeff,math.sqrt(sigeff))))
        else:
            nevent=np.random.poisson(sigeff)
        #print("%f %f %f"%(e,sigeff,nevent))
        #nevent=int(round(sigeff))
        totalevents=totalevents+nevent
        
        for j in range(nevent): 
            events.append((random.uniform(es[i][1],es[i][2]),np.random.choice(totnpix,p=np.divide(exanimap,anorm))))
    
        #print("energy now: %f events: %d"%(e,totalevents))
        #if folder and printmap and e>1500.0:
        #    hp.mollview(np.multiply(exanimap,float(totnpix)),title="anisotropy times exposure %2.1f"%e)
        #    plt.savefig("%s/expoanimap-S%d.png"%(folder,extseed),dpi=600)    
        #    printmap=False
        #totexpevents.append(sigeff)
        #poscosveladistevt.append(nb*np.dot(exanimap,poscosdistfromvela))
        #negcosveladistevt.append(-nb*np.dot(exanimap,negcosdistfromvela))
        #reales.append(e)
        #print("totalexpectedevents: ",totexpevents[-1]," poscosdistvela: ",poscosveladistevt[-1]," negcosdistvela: ",negcosveladistevt[-1])
    #cumtotexpevents=[]
    #cumposcosveladistevt=[]
    #cumnegcosveladistevt=[]
    #reve=[]
    #cte=0.0
    #cpv=0.0
    #cnv=0.0
    #for e,tev,pve,nve in zip(reversed(reales),reversed(totexpevents),reversed(poscosveladistevt),reversed(negcosveladistevt)):
    #    cte=cte+tev
    #    cpv=cpv+pve
    #    cnv=cnv+nve
    #    reve.append(e)
    #    cumtotexpevents.append(cte)
    #    cumposcosveladistevt.append(cpv)
    #    cumnegcosveladistevt.append(cnv)
    #print(e,cte,cpv,cnv,cpv-cnv,(3.0*(cpv-cnv)/cte))
    return sorted(events)


def readexmaps(extype,geom,exfolder):
    thresholds=[]
    maps=[]
    totexps=[]
    bininfo=loadresults(exfolder+"expmap-galact-%s-geo%d-bininfo.dat"%(extype,1))
    for geo in range(geom):
        n=0
        for ib in sorted(bininfo.keys()):
            if geo==0:
                thresholds.append(bininfo[ib][1])
                maps.append(loadresults(exfolder+"expmap-galact-%s-geo%d-bin%d.dat"%(extype,geo+1,ib)))
            else:
                maps[n]=np.add(maps[n],loadresults(exfolder+"expmap-galact-%s-geo%d-bin%d.dat"%(extype,geo+1,ib)))
                n=n+1
    for imap,bifi in zip(maps,sorted(bininfo.keys())):            
        totexps.append(np.sum(imap))
        #hp.mollview(imap,title="Exposure map for %2.1f - %2.1f GeV"%(bininfo[bifi][0],bininfo[bifi][1]))    
        #plt.savefig("../%s/exmaps/c-%d.png"%(folder,int(bininfo[bifi][0])),dpi=600)    
    print("exposure map thresholds: ",thresholds)
    print("total exposure list: ",totexps)
    return maps,thresholds,totexps    



def calcaniso(aniflx):
    
    rawspecs={}
    energylist=aniflx[0]
    for i in aniflx[1].keys():
        rawspecs[i]=[float(f) for f in aniflx[1][i]]
                    
    #print(rawspecs[suncoord])
    
    #print(rawspecs)
    longitudes=[]
    latitudes=[]
    distances={}
    anilist=[]
    ndisc=0
    i=0
    for ci in sorted(rawspecs.keys()):
        i=i+1
        if ci==suncoord:
            continue
        else:
            ni=False
            #print(i,ci)
            longi,lati,distance = calclonglatdist(ci)
            if (longi,lati) in distances.keys() and distance > distances[(longi,lati)][0]:
                #print("duplicate direction with larger dist:  %f %f %f"%(longi,lati,distance)) 
                ndisc=ndisc+1
                #print(i,ndisc)
                continue
            elif (longi,lati) in distances.keys() and distance < distances[(longi,lati)][0]:
                #print("duplicate direction with smaller dist:  %f %f %f"%(longi,lati,distance)) 
                ni=distance < distances[(longi,lati)][1]
                longitudes[ni]=longi
                latitudes[ni]=lati
                ndisc=ndisc+1
                #print(i,ndisc)
            else:
                longitudes.append(longi)
                latitudes.append(lati)
                distances[(longi,lati)]=(distance,i)
            if not ni:
                ni=i-ndisc            
            oposi=findoposite(ci)
            #print(" %f , %f , %f, %s"%(longi,lati,distance, str(oposi)))
            
            ani=aniso(energylist,rawspecs[ci],rawspecs[oposi],distance)
            if len(anilist)<=ni:
                anilist.append(ani)
            else:
                #print("replacing event entry ",ni) 
                anilist[ni]=ani
            #print(ani)
    return anilist,longitudes,latitudes



def diffcoeff(E):
    if lowdelt:
        return DBdiffcoeff(E,Dnorm,delt,highdelt,deltbreak,deltsoft,lowdelt,lowdeltbreak,lowdeltsoft)
    else:
        return Bdiffcoeff(E)

def Bdiffcoeff(E):
    return Dnorm*math.pow(E/DnormE,delt)/math.pow((1.0+math.pow(E/deltbreak,(delt-highdelt)/deltsoft)),deltsoft)

   
def DBdiffcoeff(R,D0,delta,deltah,RB,SP,deltal,RBl,SPl,RR=4.0):
    deltadeltal=delta-deltal
    if RR>RBl:
        D=D0*math.pow(R/RR,delta)
    else:   
        D=D0*math.pow(R/RR,deltal)
    D=D*math.pow((1+math.pow(R/RBl,math.copysign(abs(deltadeltal),RBl-RR)/SPl)),math.copysign(SPl,deltadeltal))
    deltadeltah=deltah-delta
    D=D*math.pow((1+math.pow(R/RB,math.copysign(abs(deltadeltah),RB-RR)/SP)),math.copysign(SP,deltadeltah))    
    return D	


def aniso(elist,aspec,bspec,dist):
    ani=[]
    
    for e,fa,fb in zip(elist,aspec,bspec):
        Dxx=Bdiffcoeff(e)
        dfdx=(fa-fb)/(2.0*dist*kpcincm)
        if (fa+fb)>0.0:
            ani.append((3*Dxx*dfdx/lsincmps)/(0.5*(fa+fb)))
        else:
            ani.append(0.0)
    return ani

def getclosestpoints(plo,pla,longs,lats,npoint):
    n=0
    ncl=[-1]*npoint
    mdl=[360.0]*npoint
    for lo,la in zip(longs,lats):    
        dst=angdist(plo,lo,pla,la)
        for i in range(len(ncl)):
            icl=ncl[i]
            idl=mdl[i]
            if dst<idl:
                ncl.insert(i,n)
                mdl.insert(i,dst)
                ncl.pop(-1)
                mdl.pop(-1)    
                break            
        n=n+1
    if not -1 in ncl:
        return ncl,mdl        
    else:
        print("error, required number of closest points  not found")
        sys.exit(1)

def interpolevent(events,clis,clad):
    totscale=0.0
    intevt=0.0
    for cli,cla in zip(clis,clad):
        scale=1.0/cla
        totscale=totscale+scale
        intevt=intevt+events[cli]*scale
    return intevt/totscale



def angdist(lo1,lo2,la1,la2):
    lo1=lo1*math.pi/180.0
    lo2=lo2*math.pi/180.0
    la1=la1*math.pi/180.0
    la2=la2*math.pi/180.0
    dist=(180.0/math.pi)*math.acos(math.sin(la1)*math.sin(la2)+math.cos(la1)*math.cos(la2)*math.cos(abs(lo1-lo2)))
    #print(dist)
    return dist    


def binsize(i):
   ei_min=10**((i-0.5)*0.001)
   ei_max=10**((i+0.5)*0.001)
   return ei_max-ei_min

def converttoflux(signal,exposure):
    flxdict={}
    Es=sorted(signal.keys())
    for i in range(len(Es)):
        events=signal[Es[i]]
        flx=events/binsize(i)/exposure
        flxdict[Es[i]]=[flx]
    return flxdict


def calclonglatdist(pos,sunpos=suncoord):
    dirvec=(-pos[0]+sunpos[0],-pos[1]+sunpos[1],pos[2]-sunpos[2])
    gpdir=math.sqrt(dirvec[0]**2+dirvec[1]**2)
    longi=math.atan2(dirvec[1],dirvec[0])*180.0/math.pi
    if gpdir==0.0 and dirvec[2]>0.0:
        lati=90.0
    elif gpdir==0.0 and dirvec[2]<0.0:
        lati=-90.0
    else:
        #print(dirvec[0],dirvec[2])
        lati=math.atan(dirvec[2]/gpdir)*180.0/math.pi
    dist=math.sqrt(dirvec[0]**2+dirvec[1]**2+dirvec[2]**2)
    return longi,lati,dist


def findoposite(pos,sunpos=suncoord):
    dirvec=[pos[0]-sunpos[0],pos[1]-sunpos[1],pos[2]-sunpos[2]]
    opos=[-1.0*a for a in dirvec]    
    opospos=[opos[0]+sunpos[0],opos[1]+sunpos[1],opos[2]+sunpos[2]]
    opospos=tuple([float("%1.4f"%b) for b in opospos]) 
    return opospos

def readaniso(anifilename,dragonoutpath,rawspectuple=False):
    if not rawspectuple:
        rawspecs={}
    else:
        rawspecs=rawspectuple[1]
        print("using previous rawspecs dictionary")
    anifile=open(dragonoutpath+anifilename,'r')
    strline=anifile.readline()
    #print(strline)
    ereadlist=strline.split()
    print("this should be energy: %s"%(ereadlist.pop(0)))
    print("this should be [GeV]: %s"%(ereadlist.pop(0)))
    energylist=[float(e) for e in ereadlist]
    spectype=anifile.readline().strip()
    while True:
        print(spectype)
        if spectype in ["Extra_e-","Pri_e-","Sec_e-","Sec_e+"]:
            i=0
            while True:
                sreadlist=anifile.readline().split()
                try:
                    x=float(sreadlist[0])
                except:
                    if len(sreadlist)==0:
                        spectype=""
                    else:
                        spectype=sreadlist[0]
                        print("new spectype")
                        print(spectype)
                    break
                x=float(sreadlist.pop(0))
                y=float(sreadlist.pop(0))
                z=float(sreadlist.pop(0))
                #print("%d %f %f %f"%(i,x,y,z))
                while len(sreadlist)<len(energylist):
                        sreadlist=sreadlist+[0.0]
                if not (x,y,z) in rawspecs.keys(): 
                    rawspecs[(x,y,z)]=[float(f) for f in sreadlist]
                else:
                    rawspecs[(x,y,z)]=[ float(f)+of for f,of in zip(sreadlist,rawspecs[(x,y,z)])]
                i=i+1
        elif spectype=="":
            break
        else:
            while True:
                sreadlist=anifile.readline().split()
                try:
                    x=float(sreadlist[0])
                except:
                    if len(sreadlist)==0:
                        spectype=""
                        print("end of file")
                    else:
                        spectype=sreadlist[0].strip()
                        #print(spectype)
                    break
            continue
    #print(rawspecs)
    print("complete: ",anifilename)
    return (energylist,rawspecs)


def saveresults(rfn,results):
    sfile=open(rfn,'wb')   
    pickle.dump(results, sfile, protocol=-1)
    sfile.close

def loadresults(rfn):
    sfile=open(rfn,'rb')   
    loadthing=pickle.load(sfile)
    sfile.close
    return loadthing

def main():
    snstr=""
    for SNRname in dragonSNR[1]:
        snstr=snstr+SNRname
    if sys.argv[1]=="autofolder":
        folder = "CALET-SNRtest%1.1f-%s-%s-%s"%(expofact,cond,prop,snstr)
    else:
        folder = sys.argv[1]
    os.system("mkdir "+anisodatafolder+folder)
    geo = int(sys.argv[2])
    nsampstr= sys.argv[3]
    if "-" in nsampstr:
        startnsample=int(nsampstr.partition("-")[0])
        endnsample=int(nsampstr.partition("-")[2])
    else:    
        startnsample=int(nsampstr)
        endnsample=startnsample
    for pix in m:
        hpla,hplo=hp.pix2ang(NSIDE,pix)
        hplo=hplo*(180.0/math.pi)
        if hplo>180.0:
            hplo=hplo-360.0
        hpla=90-hpla*(180.0/math.pi)
        lo[pix]=hplo
        la[pix]=hpla 
        
        #cosdistfromvela[pix]=math.cos((math.pi/180.0)*angdist(hplo,dirs[source][0],hpla,dirs[source][1]))
        #if cosdistfromvela[pix]>0.0:
        #    poscosdistfromvela[pix]=cosdistfromvela[pix]
        #elif cosdistfromvela[pix]<0.0:
        #    negcosdistfromvela[pix]=cosdistfromvela[pix]
    condtup=cond.partition("C")
    allanisoflx=False
    for SNRname in dragonSNR[1]:
        if os.path.isfile(dragonoutpath+dragonSNRfile%(SNRname,condtup[1]+condtup[2],condtup[0])):
            print("found anisotropy file")
            aniflx=readaniso(dragonSNRfile%(SNRname,condtup[1]+condtup[2],condtup[0]),dragonoutpath,allanisoflx) 
            allanisoflx=aniflx    
        else:
            print("not found: anisotropy file ",dragonoutpath+dragonSNRfile%(SNRname,condtup[1]+condtup[2],condtup[0]))
            sys.exit(1)
    
    spicklef=signalfile%(snstr,cond)
    signals=unpicklesignals(spicklef)
    infodict={}
    infodict["exmapfolder"]=exmapfolder
    infodict["geo"]=geo
    infodict["signalfile"]=signalfile
    infodict["dragonSNRfile"]=geo
    infodict["prop"]=prop
    infodict["cond"]=cond
    infodict["expofact"]=expofact
    nsample=startnsample
    es,sigeffs,exanimaps,anorms=preparemakesample(signals,aniflx,geo)
    os.system("mkdir "+anisodatafolder+folder+'/anisosamples')
    while not nsample>endnsample:
        rpicklef=anisodatafolder+folder+'/anisosamples/anisospecsample-%1.2fyr-%d.dat'%(animapnyear*float(expofact),nsample)
        print("nsample = ",nsample)
        newsample=[makesample(es,sigeffs,exanimaps,anorms,nsample)]+signals+[infodict]
        saveresults(rpicklef,newsample)    
        nsample=nsample+1    
    



if __name__ == '__main__':
    sys.exit(main())    
    

