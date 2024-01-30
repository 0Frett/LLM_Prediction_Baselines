
datasets=("pet" "airbnb")
train_mod=("img" "tab" "text" "text-tab" "img-tab" "text-img")

for dataset_name in "${datasets[@]}"; do
    for mod in "${train_mod[@]}"; do
        a="${dataset_name}/${dataset_name}_emb_img_before_trans/${mod}/train"
        b="${dataset_name}/${dataset_name}_emb_img_before_trans/${mod}/valid"
        c="${dataset_name}/${dataset_name}_emb_img_before_trans/${mod}/test"
        if [ "$dataset_name" == "airbnb" ]; then
            is_reg_option="--is_reg"
        else
            is_reg_option=""
        fi
        nohup python3 "${dataset_name}/${dataset_name}_trainclf.py" -train "$a" -valid "$b" -test "$c" -ms "classifiers/${dataset_name}/train_on_${mod}" $is_reg_option > "logs/train/${dataset_name}/${mod}.txt" &
    done
done
