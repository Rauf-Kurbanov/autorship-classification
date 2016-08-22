#! /bin/bash

PATH_TO_DIR="$1"

"$HOME"/bin/make_super_corpus.sh "$PATH_TO_DIR"/class1_training class1_training/class1_training.txt
"$HOME"/bin/make_super_corpus.sh "$PATH_TO_DIR"/class1_test class1_test/class1_test.txt
"$HOME"/bin/make_super_corpus.sh "$PATH_TO_DIR"/class2_training class2_training/class2_training.txt
"$HOME"/bin/make_super_corpus.sh "$PATH_TO_DIR"/class2_test class2_test/class2_test.txt
"$HOME"/bin/make_super_corpus.sh "$PATH_TO_DIR"/common common/common.txt
