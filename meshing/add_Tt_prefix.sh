#!/bin/bash

# Script to reinsert the Tt prefix to the lines in elem files after their loss in Matlab
# Can take wildcard arguments

for file in "${@}"; do
   if ! [[ $file =~ ".elem" ]]; then
      continue
   fi

   sed -i -e "s/^/Tt /" $file
   sed -i -r '1s/^.{3}//' $file
#   echo "$file"
done
