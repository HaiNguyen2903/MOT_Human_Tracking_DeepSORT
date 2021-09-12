search_dir='/data.local/hangd/data_vtx/mot_annotations/annotation_CH06_reviewed'

for ann_path in "$search_dir"/*.zip
    do 
        echo
        name="$(basename -- $ann_path)"
        # mkdir $search_dir/${name:0:-4}
        unzip $ann_path -d $search_dir/${name:0:-4}
    done

