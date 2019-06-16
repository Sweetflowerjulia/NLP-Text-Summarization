DATA_DIR=./data
TMP_DIR=$DATA_DIR/tmp
rm -rf $DATA_DIR $TMP_DIR
mkdir -p $DATA_DIR $TMP_DIR
# Generate data
t2t-datagen \
  --t2t_usr_dir=./train \
  --problem="onlinereview" \
  --data_dir=$DATA_DIR \
  --tmp_dir=$TMP_DIR
