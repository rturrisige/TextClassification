path='/home/rosannaturrisi/storage/NLP/'

echo 'Path to data:' $path

echo 'Starting data analysis and processing'
python data_analysis_and_processing.py $path

echo 'Extracting PCA-SciBert embedding'
python PCA_SciBert_embedding.py $path

echo 'Fine-tuning of SciBert model and embedding extraction'
python FineTuned_SciBert_embedding.py $path

echo 'Unsupervised text classification'
python text_classification.py $path

