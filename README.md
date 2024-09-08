[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org) [![Gurobi](https://img.shields.io/badge/-Gurobi-F42?style=flat&logo=gurobi&logoColor=white)](https://www.gurobi.com/)

# Modelling and solving a Collaborative Consistent Vehicle Routing Problem (CCVRP) with workload balance

This python script was developed as a project for an exam on Mathematical Optimisation. 
It is part of a study conducted on [this paper](https://www.sciencedirect.com/science/article/pii/S0377221721000035) by Simona Mancini, Margaretha Gansterer, Richard F. Hartl published on the "European Journal of Operational Research".

<br/>

## Overview
The paper presents the various mathematical constraints and objective functions for modelling and solving a centrally organized Collaborative Consistent Vehicle Routing Problem (CCVRP). 

Solving this problem means finding the optimal path that allows the carriers to complete all the deliveries to their clients. 
The collaborative part of the problem allows the carriers to exchange the customers with each other in order to minimize the usage of resources.
A client must also be served consistently according the time of the delivery and the carrier offering the service. 

The paper also proposes three heuristic algorithms to solve this problem, along with the data about their performance on a dataset shared by the authors.

<br/>

## About the Project
Our work consisted in an in-depth study of this paper and the proposed model, which was then implemented using Python and Gurobi. 
During the implementation, we cleaned up the inconsistent format of the dataset entries and experimented with the constraints and parameters. 

The three heuristic algorithms, MH, MH* and ILS, were also implemented and used to draw a comparison with the results obtained through the Gurobi optimiser and the ones reported in the paper.
The results of our tests on our updated model are available on the [project report](https://github.com/AlessandroViol/CCVPR/blob/main/Project%20report.pdf) along with the details on the constraints and parameters used.

A brief study on the scalability of this solution is available on that same report.

<br/>

## Features
Using this software the user can input any centrally organized CCVR problem and solve it using either the Gurobi solver or MH, MH* and ILS algorithms.
The solutions are then visualized through an intuitive plot.

<br/>

## Authors

- Alessandro Viol: [@AlessandrViol](https://www.github.com/AlessandroViol)
- Federica Azzalini: [@F1397](https://github.com/F1397)

<br/>

## Reference

### [The collaborative consistent vehicle routing problem with workload balance](https://www.sciencedirect.com/science/article/pii/S0377221721000035)

Simona Mancini, Margaretha Gansterer, Richard F. Hartl,

European Journal of Operational Research,
Volume 293, Issue 3,
2021,
Pages 955-965,
