import os

import numpy as np
import pandas as pd

from typing import List, Tuple, Dict

def get_coords_from_fcsv_centers_dict(pth: str,
                                 n_keys: List[str]) -> Dict[str, List[str]]:
    """
    Read the .fcsv file and extract coordinates based on the specified keys.
    
    Parameters:
    pth (str): The path to the .fcsv file.
    n_keys (List[str]): The list of keys to search for in the file.
    
    Returns:
    Dict[str, List[str]]: A dictionary with the specified keys and their coordinates.
    """
    points = {}
    with open(pth, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            for n_key in n_keys:
                if f'{n_key},' in line:
                    points[n_key] = line.split(',')[1:4]  # Get x, y, z coordinates
                    break  # Exit inner loop after finding the key
    return points

def get_coords_from_fcsv_center(pth: str,
                                n_key: str) -> dict:
    """
    Read the .fcsv file and extract coordinates based on the specified key.
    
    Parameters:
    pth (str): The path to the .fcsv file.
    n_key (str): The key to search for in the file (default is 'n').
    
    Returns:
    dict: A dictionary with the specified key and its coordinates.
    """
    points = dict()
    with open(pth, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            if f'{n_key},' in line:
                points[n_key] = line.split(',')[1:4]  # Get x, y, z coordinates
                break  # Exit loop after finding the first match
    return points

def get_coords_from_fcsv_center_list(pth: str,
                                    n_key: str) -> list:
    """
    Read the .fcsv file and extract coordinates based on the specified key.
    
    Parameters:
    pth (str): The path to the .fcsv file.
    n_key (str): The key to search for in the file.
    
    Returns:
    list: A list of coordinates [x, y, z] for the specified key, or an empty list if not found.
    """
    coordinates = []
    with open(pth, 'r') as fr:
        lines = fr.readlines()
        for line in lines:
            if f'{n_key},' in line:
                coordinates = list(map(float, line.split(',')[1:4]))  # Get x, y, z coordinates as floats
                break  # Exit loop after finding the first match
    return coordinates

def get_coordinate_system_label(fcsv_file_path: str) -> str:
    """
    Read the .fcsv file and return the coordinate system label ('LPS' or 'RAS').
    
    Parameters:
    fcsv_file_path (str): The path to the .fcsv file.
    
    Returns:
    str: 'LPS' if the coordinate system is 0, 'RAS' if it is 1, or None if not found.
    """
    with open(fcsv_file_path, 'r') as file:
        for line in file:
            if line.startswith('# CoordinateSystem'):
                # Extract the value after the equals sign and strip any extra spaces
                value = line.split('=')[1].strip()
                if value.isdigit():
                    value = int(value)
                    if value == 0:
                        return 'LPS'
                    elif value == 1:
                        return 'RAS'
    return None  # Return None if coordinate system is not found


def convert_from_lps_to_ras(lps_coords: list) -> list:
    """
    Convert LPS coordinates to RAS coordinates.

    Parameters:
    lps_coords (list): A list of LPS coordinates [x, y, z].

    Returns:
    list: A list of RAS coordinates [x, y, z].
    """
    # Define the transformation matrix for LPS to RAS
    transformation_matrix = np.array([[-1, 0, 0],  # Invert x
                                    [0, -1, 0],   # Keep y
                                    [0, 0, 1]])  # Keep z

    # Convert the LPS coordinates to a NumPy array
    lps_array = np.array(lps_coords)

    # Apply the transformation
    ras_array = transformation_matrix.dot(lps_array)

    return ras_array.tolist()


def convert_from_ras_to_lps(ras_coords: list) -> list:
    """
    Convert RAS coordinates to LPS coordinates.

    Parameters:
    ras_coords (list): A list of RAS coordinates [x, y, z].

    Returns:
    list: A list of LPS coordinates [x, y, z].
    """
    # Define the transformation matrix for RAS to LPS
    transformation_matrix = np.array([[-1, 0, 0],  # Invert x
                                    [0, -1, 0],   # Keep y
                                    [0, 0, 1]])  # Keep z

    # Convert the RAS coordinates to a NumPy array
    ras_array = np.array(ras_coords)

    # Apply the transformation
    lps_array = transformation_matrix.dot(ras_array)

    return lps_array.tolist()

def get_landmarks_labels(landmarks: List[float],
                        valids: List[int],
                        labels: List[str]) -> Tuple[List[List[float]], List[str]]:
    """
    Extracts coordinates and labels for specific landmarks based on their indices.

    Args:
        landmarks (List[float]): A flat list of landmark coordinates.
        valids (List[int]): Indices of the landmarks to be extracted.
        labels (List[str]): Labels corresponding to each landmark.

    Returns:
        Tuple[List[List[float]], List[str]]:
            - List of lists containing coordinates for the extracted landmarks.
            - List of labels corresponding to the extracted landmarks.
    """
    out_coords, out_labels = [], []
    for valid in valids:
        out_coords.append(landmarks[valid*3:valid*3+3])
        out_labels.append(labels[valid])

    return out_coords, out_labels


def fcsv_vertebrae_parser(coords: List[List[float]],
                        labels: List[str],
                        out_path: str,
                        cs = 'LPS') -> None:
    """
    Writes coordinates and labels to a file in the .fcsv format.

    Args:
        coords (List[List[float]]): Coordinates of landmarks to be written to the file.
        labels (List[str]): Labels corresponding to each landmark.
        out_path (str): Path to the output file.

    Returns:
        None: Writes data directly to the specified file.
    """
    out_file = []
    if cs == 'LPS':
        _header = '# Markups fiducial file version = 4.10\n# CoordinateSystem = 0\n# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n'
    else: #RAS
        _header = '# Markups fiducial file version = 4.10\n# CoordinateSystem = 1\n# columns = id,x,y,z,ow,ox,oy,oz,vis,sel,lock,label,desc,associatedNodeID\n'
    out_file.append(_header)

    for i, (coord, label) in enumerate(zip(coords, labels)):
        out_file.append(f'vtkMRMLMarkupsFiducialNode_{i},{coord[0]},{coord[1]},{coord[2]},0.000,0.000,0.000,1.000,1,1,0,{label},,vtkMRMLScalarVolumeNode1\n')

    with open(out_path, 'w') as ww:
        for line in out_file:
            ww.writelines(line)





