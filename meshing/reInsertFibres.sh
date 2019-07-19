#!/bin/bash

# Script to insert heart mesh with scar into torso mesh
# Usage: ./reInsertFibres.sh [scar_mesh] [torso_mesh] [submesh]

MESHTOOL=meshtool

# Check for switches and optional arguments
DEBUG=false
while getopts d flag; do
case "$flag" in
    d) DEBUG=true;;
esac
done

# Assign default values to those parameters that can be on the command line
shift $((OPTIND-1))
THISSCAR=${1:-meshes/scar_lv/myoScarLV_phi1.5708-3.1416_rho0.1-0.9_z0.3-0.9}
MESH=${2:-meshes/torso_final_ref_smooth_noAir}
SUBMESH=${3:-meshes/torso_final_ref_smooth_noAir_myo}

# Place THISSCAR lon and elem files in same folder as MESH and SUBMESH for meshtool to find
function strindex()
{
  # Function to find the last example of a substring in a string
  # Used here to find the last '/' in a filename, and hence its directory
  local x="${1%%$2*}"
  end=${1##*/}
  echo $((${#str} - ${#end}))
}
strOut=$(strindex "$MESH" /)
MESHFOLDER=${MESH:0:strOut}

if $DEBUG; then
  echo 'cp '${THISSCAR}'.lon '${MESHFOLDER}'torso_final_ref_smooth_noAir_myo.lon'
  echo 'cp '${THISSCAR}'.elem '${MESHFOLDER}'torso_final_ref_smooth_noAir_myo.elem'
else
  cp ${THISSCAR}.lon ${MESHFOLDER}torso_final_ref_smooth_noAir_myo.lon
  cp ${THISSCAR}.elem ${MESHFOLDER}torso_final_ref_smooth_noAir_myo.elem
fi

# Determine output mesh title by removing 'myo' from the myocardial scar mesh name
strOut=$(strindex "$THISSCAR" myo)
OUTMESH=${THISSCAR:0:strOut}${THISSCAR:strOut+3:${#THISSCAR}}

# Insert submesh
if $DEBUG; then
  echo $MESHTOOL' insert submesh -submsh='$SUBMESH' -msh='$MESH' -ofmt=carp_txt -outmsh='$OUTMESH' ;'
else
  $MESHTOOL insert submesh -submsh=$SUBMESH -msh=$MESH -ofmt=carp_txt -outmsh=$OUTMESH ;
fi

# Clean up by removing THISSCAR elem and lon files, and removing the newly generated .pts file (it's the same as the original!)
if $DEBUG; then
  echo 'rm '${MESHFOLDER}'torso_final_ref_smooth_noAir_myo.lon'
  echo 'rm '${MESHFOLDER}'torso_final_ref_smooth_noAir_myo.elem'
  echo 'rm ${OUTMESH}.pts'
else
  rm ${MESHFOLDER}torso_final_ref_smooth_noAir_myo.lon
  rm ${MESHFOLDER}torso_final_ref_smooth_noAir_myo.elem
  rm ${OUTMESH}.pts
fi

