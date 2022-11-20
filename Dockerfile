FROM continuumio/miniconda3

WORKDIR /GeneticAttack
COPY . .
RUN conda create --name GeneticAttack --file requirements.txt
RUN chmod -R 777 .
RUN ./setup.sh
RUN conda run -n GeneticAttack python build_embeddings.py
RUN conda run -n GeneticAttack python compute_dist_mat.py
RUN conda run -n GeneticAttack python attackRNB.py