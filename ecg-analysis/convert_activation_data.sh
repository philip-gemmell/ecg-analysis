#!/bin/bash

# Script to convert activation time data to VTK format
# Usage: ./convert_activation_data.sh [scar_mesh] [activ_dat]

#### Assign command line arguments, and adapt for derived values

# Check for presence of original mesh files, and if necessary, extract from archive files and/or
# create symbolic links to .pts file
FULL_MESH=$1
function strindex()
{
  # Function to find the last example of a substring in a string
  # Used here to find the last '/' in a filename, and hence its directory
  local x="${1%%$2*}"
  end=${1##*/}
  echo $((${#str} - ${#end}))
}
strOut=$(strindex "$FULL_MESH" /)
MESH_DIR=${FULL_MESH:0:strOut}

if [ ! -f $FULL_MESH'.elem' ]; then
	if [ ! -f $FULL_MESH'.7z' ]; then
		echo "================================"
		echo "== No valid mesh files found! =="
		echo "================================"
		return
	fi
	echo "=============================="
	echo "== Extracting mesh files... =="
	echo "=============================="
	echo "7z x -o$MESH_DIR $FULL_MESH.7z"
	7z x -o$MESH_DIR $FULL_MESH'.7z'
else
	echo "============================================="
	echo "== CARP files already present in directory =="
	echo "============================================="
fi
if [ ! -f $FULL_MESH'.pts' ]; then
	echo "==============================="
	echo "== Creating symbolic link... =="
	echo "==============================="
	echo "ln -s /home/pg16/Documents/ecg-scar/meshing/meshes/torso_final_ref_smooth_noAir.pts $FULL_MESH.pts"
	ln -s /home/pg16/Documents/ecg-scar/meshing/meshes/torso_final_ref_smooth_noAir.pts $FULL_MESH'.pts'
fi

# Derive name for extracted mesh from full mesh filename
MYO_MESH=${FULL_MESH:0:strOut}'myo'${FULL_MESH:strOut}
EXTRACT_MESH=${FULL_MESH:0:strOut}'myo'${FULL_MESH:strOut}'_extracted'


ACTIV_DAT=$2
ACTIV_VTK=${ACTIV_DAT::-4}

echo ""
echo ""
#### Extract submesh, then convert .dat file to .vtk file
{
	echo "meshtool extract myocard -msh="$FULL_MESH" -submsh="$EXTRACT_MESH" -ifmt=carp_txt -ofmt=carp_txt"
	meshtool extract myocard -msh=$FULL_MESH -submsh=$EXTRACT_MESH -ifmt=carp_txt -ofmt=carp_txt
	echo "7z u $MYO_MESH.7z ${EXTRACT_MESH}*"
	7z u $MYO_MESH'.7z' ${EXTRACT_MESH}*
	rm ${EXTRACT_MESH}*
	echo "====================================="
	echo "== meshtool extraction successful! =="
	echo "====================================="
} || {
	echo "================================="
	echo "== meshtool extraction failed! =="
	echo "================================="
	return
}

# Convert .dat file to .vtk file
echo ""
echo ""
{
	echo "GlVTKConvert -m $EXTRACT_MESH -n $ACTIV_DAT -o $ACTIV_VTK -F bin"
	GlVTKConvert -m $EXTRACT_MESH -n $ACTIV_DAT -o $ACTIV_VTK -F bin
	if [ "$?" -eq 1 ]; then
		echo "=========================="
		echo "== GlVTKConvert failed! =="
		echo "=========================="
		return
	fi	
	echo "=============================="
	echo "== GlVTKConvert successful! =="
	echo "=============================="
} || {
	echo "=========================="
	echo "== GlVTKConvert failed! =="
	echo "=========================="
	return
}
	

#### Tidy up
echo ""
echo ""
echo "rm $EXTRACT_MESH.pts"
echo "rm $EXTRACT_MESH.elem"
echo "rm $EXTRACT_MESH.lon"
rm $EXTRACT_MESH'.pts'
rm $EXTRACT_MESH'.elem'
rm $EXTRACT_MESH'.lon'
echo "========================================"
echo "== Extracted CARP mesh files removed. =="
echo "========================================"
if [ -f $FULL_MESH'.7z' ]; then
	echo "rm $FULL_MESH.pts"
	echo "rm $FULL_MESH.elem"
	echo "rm $FULL_MESH.lon"
	rm $FULL_MESH'.pts'
	rm $FULL_MESH'.elem'
	rm $FULL_MESH'.lon'
	echo "=============================="
	echo "== CARP mesh files removed. =="
	echo "=============================="
else
	echo "================================="
	echo "== No CARP mesh files removed. =="
	echo "================================="
fi
