#!/bin/bash

# based on https://github.com/quqixun/BrainPrep
if [ -n "$2" ]; then # Should get txt file with lines [paths...nii.gz tmp_path.nii.gz] and final output dir as parameters

src_paths=$1
dst_path=$2
template=template/MNI152_T1_1mm.nii.gz

echo "fslreorient2std..."
mkdir -p tmp
cat $src_paths  | parallel -C ' ' --link --jobs 6 fslreorient2std {} >> tmp/log.txt # {/} removes path

echo "1. Registration, flirt..."
mkdir -p tmp/flirt
(ls tmp/*.nii.gz | xargs -d '\n' -n 1 basename) | parallel --jobs 6 flirt -bins 256 -cost corratio -searchrx 0 0 -searchry 0 0 -searchrz 0 0 -dof 12 -interp spline -in tmp/{} -ref $template -out tmp/flirt/{} >> tmp/log.txt

echo "2. Skull-stripping, bet..."
mkdir -p tmp/bet
(ls tmp/flirt/*.nii.gz | xargs -d '\n' -n 1 basename) | parallel --jobs 6 bet tmp/flirt/{} tmp/bet/{} -R -f 0.5 -g 0  >> tmp/log.txt

echo "3. Bias Field Correction, N4BiasFieldCorrection..."
mkdir -p $dst_path
(ls tmp/bet/*.nii.gz | xargs -d '\n' -n 1 basename) | parallel --jobs 6 N4BiasFieldCorrection -i tmp/bet/{} -o $dst_path/{} -d 3 -c [100x100x60x40, 0.0001] -b 300 >> tmp/log.txt

echo "DONE"

else
echo "$0 <input directory> <output directory>"
fi


