"""Offshore measurement points and turbine outlines (longitude, latitude, region)."""

wind_offshore_pts = {
    # Measurement points in North Sea
    1001: [6.558306, 54.887932, 'North Sea'],
    1002: [6.510380, 55.455643, 'North Sea'], # North
    1003: [7.704829, 55.302157, 'North Sea'], # North-East
    1004: [7.696224, 54.774739, 'North Sea'], # East
    1005: [7.017257, 54.552354, 'North Sea'], # Center
    1006: [7.393611, 54.095704, 'North Sea'], # Near Helgoland
    1007: [5.806714, 53.808608, 'North Sea'], # South-West
    1008: [5.264095, 54.282091, 'North Sea'], # West
 
    # Measurement points in Baltic Sea
    2001: [12.325362, 54.646623, 'Baltic Sea'], # East
    2002: [12.858980, 54.819784, 'Baltic Sea'], # South
    2003: [12.912983, 55.167817, 'Baltic Sea'], # North-East
    2004: [13.647442, 55.035451, 'Baltic Sea'], # North
    2005: [14.206500, 55.156726, 'Baltic Sea'], # North-West
    2006: [14.442079, 54.634527, 'Baltic Sea'], # South-West
 
    # The remaining points outline wind farm clusters for visualization only.
    # South of the North Sea Turbines
    4: [6.109906, 54.0196156, 'Turbine'],
    5: [6.2895012, 54.0210829, 'Turbine'],
    6: [6.4241695, 53.9672452, 'Turbine'],
    7: [6.4890575, 53.9164217, 'Turbine'],
    8: [6.603899, 53.9321401, 'Turbine'],
    9: [6.7476654, 53.961489, 'Turbine'],
    10: [7.134161, 54.0110473, 'Turbine'],
    11: [7.1090126, 54.0935301, 'Turbine'],
    12: [6.821394, 54.0926241, 'Turbine'],
    13: [6.5517998, 54.0892513, 'Turbine'],
    14: [6.4155865, 54.0875648, 'Turbine'],
    15: [6.2865829, 54.0773689, 'Turbine'],
    16: [6.0788727, 54.0636698, 'Turbine'],
    17: [6.0788727, 54.0636698, 'Turbine'],
 
    # East of the North Sea Turbines
    18: [7.6413345, 54.3470519, 'Turbine'],
    19: [7.6409912, 54.5388123, 'Turbine'],
    20: [7.7783203, 54.5388123, 'Turbine'],
    21: [7.7524853, 54.4338596, 'Turbine'],
    22: [7.7663898, 54.3860556, 'Turbine'],
 
    # North of the North Sea Turbines
    23: [7.7395248, 55.0090279, 'Turbine'],
    24: [7.7412415, 55.0425323, 'Turbine'],
    25: [7.7998638, 55.0605275, 'Turbine'],
    26: [7.7999496, 54.9666769, 'Turbine'],
 
    28: [7.1704674, 55.2345539, 'Turbine'],
    29: [7.2085762, 55.2227571, 'Turbine'],
    31: [7.2464275, 55.0742397, 'Turbine'],
    32: [7.1745014, 55.0675562, 'Turbine'],
 
    33: [6.847229, 55.2853724, 'Turbine'],
    34: [6.8003654, 55.2797997, 'Turbine'],
    35: [6.9066238, 55.1268824, 'Turbine'],
    36: [6.8597603, 55.1212875, 'Turbine'],
 
    # West of the North Sea Turbines
    37: [5.7440472, 54.3041051, 'Turbine'],
    38: [6.2029839, 54.495069, 'Turbine'],
    39: [6.3198853, 54.4981595, 'Turbine'],
    40: [6.3181686, 54.5409036, 'Turbine'],
    41: [6.415844, 54.540804, 'Turbine'],
    42: [6.3682938, 54.3897541, 'Turbine'],
    43: [6.2905312, 54.3294381, 'Turbine'],
    44: [5.9098721, 54.304055, 'Turbine'],
    45: [5.9101295, 54.275048, 'Turbine'],
    46: [5.8071327, 54.2760503, 'Turbine'],
    47: [6.0182762, 54.4220262, 'Turbine'],
    48: [6.133976, 54.3294381, 'Turbine'],
 
    # Baltic Sea Turbines
    481: [13.9883423, 54.9199405, 'Turbine'],
    49: [14.0302277, 54.9214204, 'Turbine'],
    50: [14.1768265, 54.8097872, 'Turbine'],
    51: [14.1397476, 54.7531612, 'Turbine'],
    52: [14.0322876, 54.8012786, 'Turbine'],
 
    53: [13.9214802, 54.7892051, 'Turbine'],
    54: [13.8972759, 54.8606038, 'Turbine'],
    55: [13.815136, 54.8565033, 'Turbine'],
    56: [13.801403, 54.8139419, 'Turbine'],
 
    57: [13.7432098, 54.7884133, 'Turbine'],
    58: [13.6129189, 54.8535882, 'Turbine'],
    59: [13.5730934, 54.8299633, 'Turbine'],
 
    60: [13.0838585, 54.9496254, 'Turbine'],
    61: [13.1516647, 55.0052871, 'Turbine'],
    62: [13.228569, 55.008585, 'Turbine'],
    63: [13.198185, 54.9772689, 'Turbine'],
 
    64: [12.6299858, 54.6109012, 'Turbine'],
    65: [12.6918697, 54.6306797, 'Turbine'],
    66: [12.6299858, 54.5839512, 'Turbine'],
}
 
# Keep only measurement points, not turbine outline points.
SEA_POINTS = {
    pid: coords
    for pid, coords in wind_offshore_pts.items()
    if coords[2] in ('North Sea', 'Baltic Sea')
}
