FROM continuumio/miniconda3

WORKDIR /GeneticAttack
COPY . .
RUN conda env create -f environment.yml
RUN chmod -R 777 .
RUN ./setup.sh
RUN conda run -n GeneticAttack python build_embeddings.py
RUN conda run -n GeneticAttack python compute_dist_mat.py
CMD [ "conda","run","-n","GeneticAttack","python","attackRNB.py"]
CMD [ "conda","run","-n","GeneticAttack","python","attackMNB.py"]