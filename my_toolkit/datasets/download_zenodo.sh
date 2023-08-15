DOI=$1
DIR=$2

bash -c "zenodo_get -d $DOI -o $DIR && echo 'Unzipping...' "
bash -c "find $DIR -type f -name '*.tgz' -exec tar -xvzf '{}' -C $DIR \; -exec rm '{}' \;"
bash -c "echo '🚀 Done!' "