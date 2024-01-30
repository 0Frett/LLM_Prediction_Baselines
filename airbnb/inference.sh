
datasets=("airbnb" "pet")
train_mod=("text-tab") #"img" "tab" "text" "img-tab" "text-img" "text-tab"
test_mod=("img")

#test transfer
method="DDRM" #SDEdit"DDRM"
for dataset_name in "${datasets[@]}"; do
    for mod1 in "${train_mod[@]}"; do
        for mod2 in "${test_mod[@]}"; do
            a="trans/${dataset_name}/${method}/train_${mod1}_test_${mod2}"
            b="classifiers/${dataset_name}/train_on_${mod1}"
            c="train_${mod1}_test_${mod2}"
            nohup python3 "${dataset_name}/${dataset_name}_inference.py" -test "$a" -m "$b" -ms "$c" > "logs/inference/${dataset_name}/${method}/${c}.txt" &
        done
    done
done

# test without transfer
# for dataset_name in "${datasets[@]}"; do
#     for mod1 in "${train_mod[@]}"; do
#         for mod2 in "${test_mod[@]}"; do
#             a="${dataset_name}/${dataset_name}_emb_img_before_trans/${mod2}/test"
#             b="classifiers/${dataset_name}/train_on_${mod1}"
#             c="NONTRANS_train_${mod1}_test_${mod2}"
#             nohup python3 "${dataset_name}/${dataset_name}_inference.py" -test "$a" -m "$b" -ms "$c" > "logs/inference/${dataset_name}/${c}.txt" &
#         done
#     done
# done