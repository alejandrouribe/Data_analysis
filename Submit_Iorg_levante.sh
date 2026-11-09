"""
This script calculates monthly results by modifying the Iorg python base script and submit the jobs to slurm.
Script used in the German supercomputer Levante for the NextGEMS hackathon Vienna, 2022.
@author: Alejandro UC
"""
#!/bin/bash
#-----------------------------------------------------------------------------
## model name
#-----------------------------------------------------------------------------
model='ngc2009'
#-----------------------------------------------------------------------------
## Looping along time (in months)
#-----------------------------------------------------------------------------
for i in {0..25}
do
time_0=$(date +%Y-%m-%d -d "2020-01-01 +$i months")
j=$(expr $i + 1)
time_1=$(date +%Y-%m-%d -d "2020-01-01 +$j months -1 days")
echo $time_0, $time_1
#-----------------------------------------------------------------------------
## If output file already exists, continue to next month
#-----------------------------------------------------------------------------
if test -f "/work/bb1153/m300648/NextGEMS/outdata/Iorg_${model}_${time_0}_${time_1}.nc"
then
echo "/work/bb1153/m300648/NextGEMS/outdata/Iorg_${model}_${time_0}_${time_1}.nc FILE exists."
continue
#-----------------------------------------------------------------------------
## Else create a Iorg python script for that specific month
#-----------------------------------------------------------------------------
else
echo "/work/bb1153/m300648/NextGEMS/outdata/Iorg_${model}_${time_0}_${time_1}.nc does not exist."
sed "s/time_0='2020-01-21'/time_0='${time_0}'/g; s/time_1='2020-01-23'/time_1='${time_1}'/g" IORG_rlut_NG_hack.py > IORG_rlut_NG_${model}_${time_0}_${time_1}.py
#-----------------------------------------------------------------------------
## Write a SLURM batch script
#-----------------------------------------------------------------------------
cat << EOF > 2run_${model}_${time_0}_${time_1}.sh
#!/bin/bash
#SBATCH --partition=shared
#SBATCH --account=bb1153
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH --output=${model}_${time_0}_${time_1}.out
#SBATCH --mail-type=FAIL
module load python3
module load cdo
python3 IORG_rlut_NG_${model}_${time_0}_${time_1}.py >  IORG_rlut_NG_${model}_${time_0}_${time_1}.out
EOF
#-----------------------------------------------------------------------------
## Submit the job
#-----------------------------------------------------------------------------
sbatch 2run_${model}_${time_0}_${time_1}.sh
fi
done