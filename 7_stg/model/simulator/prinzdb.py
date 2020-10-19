""" 
    Usage:
        
        import prinzdb

        # This loads the biological parameters located in the folder 'data_dir'
        # The data can be found in the folder 'pyloric-network-simulator' which
        # can be downloaded from the following website:
        #   
        # http://www.biology.emory.edu/research/Prinz/database-sensors/
        #
        prinzdb.load_params(data_dir)

        snapshots = {}
        # Load the snapshots from the given file into the provided dictionary
        # Snapshots are indexed by their network codes
        prinzdb.load_snapshots(data_dir, filename, snapshots)

        # This function converts the saved network into the format used by
        # the HH simulator (neurons, synaptic connections & initial values)
        models, conns, init = prinzdb.create_model(netcode, snapshots)
    
        # Now you can run the simulator with the provided data

"""

import numpy as np

import HH

import pickle
import sys

# The names of the neurons and the synapses in the order appearing in Prinz' codebase
# These are used to locate the files containing the relevant data
neutypes = [ 'PM', 'LP', 'PY' ]
syntypes = [ 'ABLP', 'PDLP', 'ABPY', 'PDPY', 'LPPM', 'LPPY', 'PYLP' ]

# These are the names of additional files to be opened
paramkeys = [ ]

paramdata = {}
syndata = {}
neudata = {}

def build_conns(params):
    return np.asarray([ [ 1, 0, params[0], HH.Esglut, HH.kminusglut ], \
                        [ 1, 0, params[1], HH.Eschol, HH.kminuschol ], \
                        [ 2, 0, params[2], HH.Esglut, HH.kminusglut ], \
                        [ 2, 0, params[3], HH.Eschol, HH.kminuschol ], \
                        [ 0, 1, params[4], HH.Esglut, HH.kminusglut ], \
                        [ 2, 1, params[5], HH.Esglut, HH.kminusglut ], \
                        [ 1, 2, params[6], HH.Esglut, HH.kminusglut ] ])

def load_params(data_dir):
    for k in paramkeys:
        data = []
        inp = open("{}/{}.dat".format(data_dir, k), "r")
        for l in inp.readlines():
            if l.isspace():
                continue

            data.append(float(l))

        paramdata[k] = data

    print("num neutypes:  ", len(neutypes))
    for k in neutypes:
        data = []
        inp = open("{}/{}_Gs.dat".format(data_dir, k), "r")
        for l in inp.readlines():
            if l.isspace():
                continue

            parts = l.split(' ')

            floats = [ float(p) * 1e3 for p in parts ]
            data.append(floats)


        neudata[k] = data

    print("num syntypes:  ", len(syntypes))
    for s in syntypes:
        data = []
        inp = open("{}/{}_strengths.dat".format(data_dir, s), "r")
        for l in inp.readlines():
            if l.isspace():
                continue

            data.append(float(l) * 1e-6)        # We use mS, they use nS

        syndata[s] = data

def load_tododata(data_dir, filename):
    """ 
        Loads the given file and returns a list whose entries are the lines of
        the file (the format is that of the file '100todo.dat')
    """

    ret = []

    todoinp = open("{}/{}.dat".format(data_dir, filename), "r")
    for l in todoinp.readlines():
        if l.isspace():
            continue

        parts = l.split(' ')

        netnum = int(parts[0]) * 10000 + int(parts[1])
        netcode = parts[2].zfill(5) + parts[3].zfill(5)
        savedconn = int(parts[4])
        savedperiod = float(parts[5])

        data = { 'netnum' : netnum, 'netcode' : netcode, 'savedconn' : savedconn, 'savedperiod' : savedperiod }

        ret.append(data)

    return ret

# Most of this is self-explanatory with the help of Prinz' documentation
def readsnapshot(line):
    inp = line.split(" ")

    netnum1 = int(inp[0])
    netnum2 = int(inp[1])

    netnum = netnum1 * 10000 + netnum2

    # The network code is a 10-digit integer split into two halves (must use zero padding)
    netcode1 = inp[2].zfill(5)
    netcode2 = inp[3].zfill(5)

    netcode = "{}{}".format(netcode1, netcode2)

    # We order the m and h variables differently, so we have to rearrange elements
    translator = [ 0, 1, 2, 4, 6, 8, 10, 11, 12, 3, 5, 7, 9 ]

    neuvars = {}
    neudata = inp[4:4 + len(neutypes) * len(translator)]
    neudata = np.asarray([ float(x) for x in neudata ]).reshape(-1,len(translator))
   
    for k, d in zip(neutypes, neudata):
        d[0] *= 1e3     # We use mV for the membrane potential
        d[1] *= 1e6     # We use muM for the Ca concentration
        neuvars[k] = [ d[translator[i]] for i in range(len(translator)) ]

    syndata = inp[4 + len(neutypes) * 13:]
    synstrs = [ float(x) for x in syndata ]

    return { 'netnum' : netnum, 'netcode' : netcode, \
             'neuvars' : neuvars, 'synstrs' : synstrs }

def load_snapshots(data_dir, filename, out):
    datfile = open("{}/{}".format(data_dir, filename), "r")

    while True:
        l = datfile.readline()
        if l == '':
            break

        snapshot = readsnapshot(l)
        netcode = snapshot["netcode"]
        out[netcode] = snapshot

def create_model(netcode, snapshots = None):
    assert(len(netcode) == len(neutypes) + len(syntypes))

    if snapshots == None or netcode not in snapshots.keys():
        init_list = None
    else:
        # Create init data from snapshot
        snapshot = snapshots[netcode]

        init_entries = {}

        for k in neutypes:
            init_entries[k] = snapshot["neuvars"][k]

        for i in range(len(syntypes)):
            init_entries[syntypes[i]] = snapshot['synstrs'][i]

        init_list = [ init_entries[k] for k in neutypes ] + [ [init_entries[s] for s in syntypes] ]

    # Retrieve neuron and synapse types (indexed by integers)

    neuinds = {}

    for k, i in zip(neutypes, netcode[:len(neutypes)]):
        neuinds[k] = int(i)

    syninds = {}

    for s, i in zip(syntypes, netcode[len(neutypes):]):
        syninds[s] = int(i)

    # Retreive neuron models and synapse connection strengths
    neumodels = {}
    for k in neutypes:
        neumodels[k] = neudata[k][neuinds[k]]

    g_synmaxs = {}

    for s in syntypes:
        g_synmaxs[s] = syndata[s][syninds[s]]

    # Return the correct setup data
    conns = [ [ 1, 0, -g_synmaxs["ABLP"], HH.Esglut, HH.kminusglut ], \
              [ 1, 0, -g_synmaxs["PDLP"], HH.Eschol, HH.kminuschol ], \
              [ 2, 0, -g_synmaxs["ABPY"], HH.Esglut, HH.kminusglut ], \
              [ 2, 0, -g_synmaxs["PDPY"], HH.Eschol, HH.kminuschol ], \
              [ 0, 1, -g_synmaxs["LPPM"], HH.Esglut, HH.kminusglut ], \
              [ 2, 1, -g_synmaxs["LPPY"], HH.Esglut, HH.kminusglut ], \
              [ 1, 2, -g_synmaxs["PYLP"], HH.Esglut, HH.kminusglut ] ]

    models_list = [ neumodels["PM"], neumodels["LP"], neumodels["PY"] ]

    return models_list, conns, init_list
