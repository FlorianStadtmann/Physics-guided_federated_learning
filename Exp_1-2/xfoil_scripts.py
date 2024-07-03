import numpy as np
import subprocess
import os
import turbine_data as turbine

def run_xfoil(Re=1e7,       # Reynolds number
              Mach=0.2,     # Mach number
              filt=3,       # Number of times to apply Hannold's filter
              alpha=5,      # One or a range of angle of attack
              iter=500,     # Max number of iterations in xfoil's solver
              airfoil = "AF20.txt",             # Airfoil shape file name
              working_directory = "xfoil_work",
              timeout = 5):    # How long to wait before cancel
    '''Runs xfoil with a single airfoil, single Mach and Reynolds, single or multiple angles of attack.
    Exceptions:
        TimeoutError: xfoil could not produce a result.
        ValueError: xfoil did not converge.
    Returns dictionary with all fields from xfoil polar accumulation file (lists):
        "alpha", "CL", "CD", "CDp", "CM", "Top_Xtr", "Bot_Xtr"'''
    
    # Allows using airfoil integers
    if type(airfoil) == int:
        airfoil = "AF"+str(airfoil).zfill(2)+".txt"
    
    # If xfoil is left running by a previous instance.
    os.system("taskkill /im xfoil.exe /F")
    
    # Determine paths
    inputfile = os.path.join(working_directory,"xfoil_commands.txt")
    if os.path.exists(inputfile):   os.remove(inputfile)
    airfoil_path = os.path.join(working_directory,airfoil)
    outputfile = working_directory+"\\"+airfoil.rstrip(".txt")+".polar"

    # Input data validity check
    # assert os.path.exists(airfoil_path)
    # assert isinstance(Re,(int,float))
    # assert isinstance(Mach,(int,float))
    # assert isinstance(filt,int)
    # assert isinstance(iter,int)
    # assert isinstance(alpha,(int,float,list))
    # if isinstance(alpha,list): assert len(alpha) in [1,3]
    
    # Write xfoil commands
    commands = []
    commands.append("PLOP")
    commands.append("G F")
    commands.append("")
    commands.append("LOAD "+airfoil_path)
    commands.append(airfoil.rstrip(".txt"))
    commands.append("MDES")
    for i in range(filt): commands.append("FILT")
    commands.append("")
    commands.append("OPER")
    commands.append("VISC "+str(Re))
    commands.append("MACH "+str(Mach))
    commands.append("ITER "+str(iter))
    commands.append("PACC")
    commands.append(outputfile)
    commands.append("")
    if isinstance(alpha,list) and len(alpha)==3:
        commands.append("aseq "+" ".join([str(a) for a in alpha]))
    if isinstance(alpha,list) and len(alpha)==1:
        commands.append("a "+str(alpha[0]))
    if isinstance(alpha,(int,float)):
        commands.append("a "+str(alpha))
    commands.append("")
    commands.append("quit")
    
    # Write commands to file and run xfoil
    with open(inputfile,"w") as f:  f.write('\n'.join(commands))
    if os.path.exists(outputfile): os.remove(outputfile)
    order = working_directory+"\\xfoil.exe < "+inputfile
    
    try:
        # p = subprocess.Popen(order,shell=True,start_new_session=True)
        # p.wait(timeout=timeout)
        subprocess.run(order,shell=True,timeout=timeout)
    except subprocess.TimeoutExpired:
        os.system("taskkill /im xfoil.exe /F")
        raise TimeoutError
        
    # Read polar output file
    with open(outputfile,"r") as f:
        lines = f.readlines()
        lines = [l.lstrip(" ").rstrip("\n") for l in lines]
        lines = lines[12:]
        N = len(lines)
        if N==0: raise ValueError
        results = { "alpha": np.empty(N),
                    "CL": np.empty(N),
                    "CD": np.empty(N),
                    "CDp": np.empty(N),
                    "CM": np.empty(N),
                    "Top_Xtr": np.empty(N),
                    "Bot_Xtr": np.empty(N)}
        for l in range(N):
            keys = list(results.keys())
            vals = np.array(lines[l].split("  "),dtype=float)
            for v in range(len(vals)):
                results[keys[v]][l] = vals[v]
    results["airfoil"] = airfoil.rstrip(".txt")
    
    return results

def camber_and_thickness(airfoil = 20,
                        working_directory = "xfoil_work/"):
    '''Calculate camber rating and thickness rating of an airfoil shape.
    Returns:
        camber: int
        thickness: int 
        shape: dictionary - containing both surfaces and camber line.'''
    
    x,y = turbine.get_xy_from_file(airfoil)
    
    # Define upper and lower camber
    m = 100
    xlower = x[:m]
    xupper = x[m:]
    ylower = y[:m]
    yupper = y[m:]

    # initialize
    camber = np.zeros((2,m))
    chordl = np.zeros((2,m))
    chordl[0] = xlower
    chordl[1] = 0

    # Find camber line: Midpoint between upper and lower camber
    for i in range(m):
        xli = xlower[i]
        yli = ylower[i]
        
        nearest = np.argmin(np.abs(xupper-xli))
        xui = xupper[nearest]
        yui = yupper[nearest]

        camber[0][i] = xli
        camber[1][i] = (yui+yli)/2

    # Camber rating: Ratio of maximum chord-camber distance to chord length
    camber_rating = np.max(np.abs(camber[1]))
    
    # Thickness rating: Ratio of maximum thickness to chord length
    thickness_rating = (np.max(y)-np.min(y))/(np.max(x)-np.min(x))
    
    shape = {"xlower":xlower,"xupper":xupper,"ylower":ylower,"yupper":yupper,"camber":camber}
    
    return camber_rating,thickness_rating, shape

def analyze_airfoil(Re=1e7,Mach=0.2,filt=3,alpha=5,iter=500,airfoil = "AF20.txt",timeout=10,
        inputfile="input_commands.txt",working_directory="xfoil_work"):
    '''Returns dictionary like run_xfoil, with added camber and thickness values.'''
    
    data = run_xfoil(Re,Mach,filt,alpha,iter,airfoil,working_directory,timeout)
    
    cam,thi,shape = camber_and_thickness(airfoil,working_directory)
    
    data["camber"] = cam
    data["thickness"] = thi
    data["shape"] = shape
    
    return data