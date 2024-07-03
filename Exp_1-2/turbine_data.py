import numpy as np
import os

def airfoil_shape_path(n):
    '''Returns relative airfoil shape path, given airfoil number [0,29].'''
    foildir = "C:/data/Reference_Turbines/IEA-10.0-198-RWT-master/openfast/Airfoils"
    airfoils = os.listdir(foildir)
    airfoils = [os.path.join(foildir,x) for x in airfoils if x.endswith("Coords.txt")]
    return airfoils[n]

def airfoil_polar_path(n):
    '''Returns relative airfoil polar path, given airfoil number [0,29].'''
    foildir = "C:/data/Reference_Turbines/IEA-10.0-198-RWT-master/openfast/Airfoils"
    polars = os.listdir(foildir)
    polars = [os.path.join(foildir,x) for x in polars if "Polar" in x]
    return polars[n]

def get_notrail_xy(airfoil=20,clockwise=True):
    '''Returns x and y coordinates of plain text file containing coordinates of notrail airfoil shape produced by xfoil, in clockwise order by default.'''
    airfoil = 20
    filename = "xfoil_work/"+"AF"+str(airfoil)+"notrail.txt"
    with open(filename) as file:
        lines = file.readlines()
    x = []
    y = []
    for line in lines:
        line = line.replace("\n","").split(" ")
        line = [float(l) for l in line if l!=""]
        x.append(line[0])
        y.append(line[1])
    if clockwise:
        return np.array(x),np.array(y) 
    else:
        return np.array(x)[::-1],np.array(y)[::-1]

def get_xy_from_file(airfoil=20,clockwise=True):
    '''Returns x and y coordinates as two lists given openfast airfoil number [0,29], in clockwise order by default.'''
    filename = airfoil_shape_path(airfoil)
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    lines = lines[8:]
    x = []
    y = []
    for line in lines:
        l = line.split(" ")
        l = [x for x in l if x!='']
        x.append(float(l[0]))
        y.append(float(l[1]))
    if clockwise:
        return np.array(x),np.array(y) 
    else:
        return np.array(x)[::-1],np.array(y)[::-1]

def get_aots():
    '''Returns list of angles of attack used in dataset (alphas)'''
    alpha,_,_,_ = get_ClCd_from_file()
    return alpha

def get_ClCd_from_file(airfoil=20):
    '''Returns IEA Cl, Cd, Cm coefficients from file.'''
    filename = airfoil_polar_path(airfoil)
    with open(filename) as file:
        lines = file.readlines()
        lines = [line.rstrip() for line in lines]
    lines = lines[54:]
    Cl = []
    Cd = []
    Cm = []
    alpha = []
    for line in lines:
        l = line.split(" ")
        l = [x for x in l if x!='']
        alpha.append(float(l[0]))
        Cl.append(float(l[1]))
        Cd.append(float(l[2]))
        Cm.append(float(l[3]))
    return np.array(alpha), np.array(Cl), np.array(Cd), np.array(Cm)

def get_speed(airfoil=-1,rpm=7,Mach=False):
    '''Get air speed of airfoils based on distance from center and rpm.
    If airfoil in [0,29]: float (one airfoil).
    If airfoil = -1: list (all 30 airfoils)
    Mach: return mach speed instead of m/s.'''

    span = get_blade_data()["BladeSpan"]
    if airfoil != -1: span = span[airfoil]
    
    speed = rpm * (1/60) * (span+2.3) * (2*np.pi)
    return speed/338 if Mach else speed

def get_blade_data():
    '''Loads IAE blade shape definition data'''
    dir = r"../IEA-10.0-198-RWT/openfast/"
    file = os.path.join(dir,"IEA-10.0-198-RWT_AeroDyn15_blade.dat")
    with open(file,'r') as f:
        lines = f.read()
    lines = lines.split("\n")
    lines = lines[6:36]
    lines = [l.split(" ") for l in lines]
    lines = [[l for l in line if l!=''] for line in lines]

    data = np.zeros((6,30))

    for i in range(30):
        for j in range(6):
            data[j][i] = float(lines[i][j])
    
    return {"BladeSpan": data[0],
            "BladeCurvature": data[1],
            "BladeSweep": data[2],
            "BladeCurvatureAngle": data[3],
            "BladeTwistAngle": data[4],
            "BladeChordLength": data[5]}

def get_reduced_resolution_shape(airfoil=20, resolution=0.1, method="simplified"):
    '''Given airfoil number, and a resolution fraction, attempts to give a reduced-resolution
    representation of the airfoil. 
    If method=="simplified": Returns regular (x,y) coordinate lists.
    If method=="divided": Returns upper, camber, and lower surface.
    
    '''
    
    x,y = get_xy_from_file(airfoil)
    pts = int(100*resolution)
    interval = int(1/resolution)

    if method=="simplified":
        rx = []
        ry = []
        
        for i in range(len(x)):
            if i%interval==0:
                rx.append(x[i])
                ry.append(y[i])
        return np.array(rx), np.array(ry)

    m = 100
    xlower = x[:m]
    xupper = x[m:]
    ylower = y[:m]
    yupper = y[m:]

    lower = np.zeros((2,pts))
    upper = np.zeros((2,pts))
    camber = np.zeros((2,pts))

    for i in range(m):
        if i%interval==0:
            j = int(i/interval)
            xli = xlower[i] # x of lower point
            yli = ylower[i] # y of lower point
            nearest = np.argmin(np.abs(xupper-xli))
            xui = xupper[nearest]
            yui = yupper[nearest] # y of upper point closest in the x-dimension

            camber[0][j] = xli
            camber[1][j] = (yui+yli)/2
            lower[0][j] = xli
            lower[1][j] = yli
            upper[0][j] = xui
            upper[1][j] = yui
    return upper, camber, lower

def IEAreport_airfoil_polar_read(airfoil=0):
    '''(DEPRECATED) Reads polar file from IEA report'''
    dir = "IEA_report_airfoils"
    file = os.path.join(dir,os.listdir(dir)[airfoil])
    with open(file,"r") as f:
        lines = f.read()
    lines = lines.split("\n")
    lines = [l.split(" ") for l in lines]
    alpha = []
    Cl = []
    Cd = []
    Cm = []
    for i in range(len(lines)):
        alpha.append(float(lines[i][0]))
        Cl.append(float(lines[i][1]))
        Cd.append(float(lines[i][2]))
        Cm.append(float(lines[i][3]))
    for i in range(len(lines)):
        alpha.append(float(lines[i][4]))
        Cl.append(float(lines[i][5]))
        Cd.append(float(lines[i][6]))
        Cm.append(float(lines[i][7]))
    for i in range(len(lines)-2):
        alpha.append(float(lines[i][8]))
        Cl.append(float(lines[i][9]))
        Cd.append(float(lines[i][10]))
        Cm.append(float(lines[i][11]))
    return alpha, Cl, Cd

