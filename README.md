# LLM_Prediction_Baselines

### Datasets
The code in this repository is designed to work with datasets: Airbnb and PetFinder. The embedding-images before transfer is transformed by _create_img.sh_ and are saved in each dataset folder.



### Baselines
The code in this repository is designed to work with Baselines: SDEdit and DDRM. We pretrained score-based generative models using the DDIM framework for each modality and both baseline models leverage the same pretrained model. The embedding-images after transfer are saved in the _trans_ folder.  
 


### Results
Both baselines shared the same downstream model, which are saved in the _classifiers_ folder. Inference results are saved in the _logs&result_. Used _train_clf.sh_ and _inference.sh_ to train downstream model and inference.

### Visualization
Visualization of embeddings transfer using UMAP for several combinations are saved in _visualization_ folder. Used _visualize.sh_ to visualize other modalities transfer. 

### Hyperparameters
See the _hyperparameters_ folder for baseline configs and args settings.

