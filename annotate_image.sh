#!/bin/bash

# Script to annotate series of text files

i=0
for filename in "${@}"; do
	filename_text=${filename/.png/_text.png}
	cmd='convert -font helvetica -fill blue -pointsize 36 -draw "text 15,50 '$i'" '$filename' '$filename_text
	echo $cmd
	#eval $cmd
	((i++))
done
