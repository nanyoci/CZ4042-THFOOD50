# How to Run the Experiments
1. `conda create --name ass2_vit` (create a virtual environment)
2. `conda activate ass2_vit`
3. `conda install pip`
4. `pip install -r requirements.txt`
5. `pip freeze > requirements.txt` (save the requirements with the version numbers for submission)
6. `sbatch food_job.sh` (maybe can send me Food_Output file to see if the output is ok, cuz might need to adjust parameters or something)
7. `sbatch original_job.sh`
8. `sbatch no_pretrain_job.sh`
9. Send me all the output, err & model folders (if the model folders are too big just send me the img)
