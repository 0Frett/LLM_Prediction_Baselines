datasets=("airbnb" "pet")

for dataset_name in "${datasets[@]}"; do
    nohup python3 "${dataset_name}/${dataset_name}_create_img.py" --text_ok --img_ok > logs/gen_img/${dataset_name}/text-img.txt &
    nohup python3 "${dataset_name}/${dataset_name}_create_img.py" --text_ok --tabular_ok > logs/gen_img/${dataset_name}/text-tabular.txt &
    nohup python3 "${dataset_name}/${dataset_name}_create_img.py" --img_ok --tabular_ok > logs/gen_img/${dataset_name}/img-tabular.txt &
    nohup python3 "${dataset_name}/${dataset_name}_create_img.py" --text_ok > logs/gen_img/${dataset_name}/text.txt &
    nohup python3 "${dataset_name}/${dataset_name}_create_img.py" --img_ok > logs/gen_img/${dataset_name}/img.txt &
    nohup python3 "${dataset_name}/${dataset_name}_create_img.py" --tabular_ok > logs/gen_img/${dataset_name}/tabular.txt &
done