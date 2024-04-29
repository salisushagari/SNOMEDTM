# SNOMEDTM
A transformer based model for Adverse drug event extraction
follow the following link to download the remaining file (larger size file) https://drive.google.com/drive/folders/1xITmjCTH6C8xCWkYUoz-jZJLIEP7yGxh?usp=drive_link
To fine-tune this model for Adverse drug event task, we adopted a multi-task learning a dual sequence modelling system MTTLADE system (El-alaly et al., 2021) from the following repository https://github.com/drissiya/MTTLADE. Add the model as the new model task_def.py file and used it to create shared represention using Hugging Face AutoModel from pre-trained to initialize the parameters of the model to benefit from the pre-trained parameters from medical terminologies of SNOMED-CT and MedDRA.
