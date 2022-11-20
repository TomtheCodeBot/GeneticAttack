#!/bin/bash
docker build . -t geneticattack
docker run -v  `pwd`/mydata:/GeneticAttack --name attacking  -d geneticattack