#!/bin/zsh

print_usage() {
    printf "run octomap.\n"
    printf "inputs:\n"
    printf "    -d <dataset_folder>\n"
    printf "    -s <seq_name1> <seq_name2> ...\n"
}
WARN='\033[0;33m'
OK='\033[0;32m'
NC='\033[0m'

folder_set=false
sequence_set=false

while getopts "d:s:" flag; do
    case ${flag} in
    d)
        DATASET_FOLDER=${OPTARG}
        folder_set=true
        ;;
    s)
        process_list=("$OPTARG")
        until [[ $(eval "echo \${$OPTIND}") =~ ^-.* ]] || [ -z $(eval "echo \${$OPTIND}") ]; do
            process_list+=($(eval "echo \${$OPTIND}"))
            OPTIND=$((OPTIND + 1))
        done
        sequence_set=true
        ;;
    esac
done

[ $folder_set = false ] && print_usage && exit 1
[ $sequence_set = false ] && print_usage && exit 1

source /opt/ros/noetic/setup.zsh
source catkin_ws/devel/setup.zsh

for seq in $process_list; do

    seq_folder=$DATASET_FOLDER/$seq
    if [ ! -d $seq_folder ]; then
        printf "$WARN\n"
        echo "folder does not exist: $seq_folder $NC"
        continue
    fi

    bfs=""
    for bf in $seq_folder/bags_w_radar/*; do bfs=${bfs}\ $bf; done

    roslaunch radar_map octomap.launch \
        dataset_path:=$seq_folder \
        bag_files:="$bfs"

done
