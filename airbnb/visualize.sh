dataset_name="airbnb" #"airbnb"
method="DDRM" #"SDEdit"

text2image="trans/${dataset_name}/${method}/train_img_test_text"
img2tab="trans/${dataset_name}/${method}/train_tab_test_img"
text="${dataset_name}/${dataset_name}_emb_img_before_trans/text/test"
img="${dataset_name}/${dataset_name}_emb_img_before_trans/img/test"
tab="${dataset_name}/${dataset_name}_emb_img_before_trans/tab/test"

python3 visualize.py --data_name $dataset_name --method $method --text2image $text2image --img2tab $img2tab --text $text --img $img --tab $tab
