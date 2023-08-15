DATASET=$1
DIR=$2

if [ $DATASET = 'SM' ]
then
source download_zenodo.sh 10.5281/zenodo.3685861 $DIR
elif [ $DATASET = 'BSM' ]
then
source download_zenodo.sh 10.5281/zenodo.6772776 $DIR
else
echo "Invalid dataset name - please choose from: SM, BSM"
fi