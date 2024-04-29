# SNOMEDTM
A transformer based model for Adverse drug event extraction
To fine-tune this model for Adverse drug event task we adopted a dual sequence modelling system MTTLADE system from the following repository. Add the model as the new model and used it to create shared represention using Hugging Face AutoModel from pre-trained to initial the parameters of the to benefit from the pre-trained parameters from medical terminologies of SNOMED-CT and MedDRA.
