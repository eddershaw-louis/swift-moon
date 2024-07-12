import swiftsimio as sw
import inspect
import os

file = sw.load("hotel_impact_spinning_0.7M_b30_r168_0240.hdf5")
help(file.gas.smoothing_lengths.convert_to_mks)
#print(os.path.abspath(inspect.getfile(file.gas.smoothing_lengths.convert_to_mks())))
