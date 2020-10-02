# Welcome to our CS 4641 Project
Group Members:
Daniel Mulloy, Steven McGaughey, William Hunnicutt, and Hunter Copp.

## Touch-points
### Touch-point 1 (September 28)
[Google Slides proposal](https://docs.google.com/presentation/d/1lqc4cYwl3FGDUaEJnqRbJutyHcS9bqcUW0vrNTv1BoU/edit?usp=sharing)  
[Pre-recorded video of proposal presentation](assets/proposal.mp4)

### Touch-point 2 (October 30)


### Touch-point 3 (November 20)

## Project proposal (October 2)

### Summary Figure
![Infographic](./assets/infographic.png)

### Introduction/Background
Problem we aim to address, motivation, and goal  
Problem: The outcome of a football play is considered to be very unpredictable. The factors that go into it (weather, positions, remaining yards, etc) make it hard for a coach to judge in the moment which plays will allow the team to get the yards needed to succeed.  
Motivation: Being able to model the number of yards that will be gained by a play will yield a team a competitive advantage.  
Goal: To model the number of yards that will be gained in an NFL play based on the situation.  
Disclaimer: For simplicity, we will not be including field goals, and for punts, we will model the number of yards gained after the ball is recieved.

### Dataset 
For our project, we plan on using a [play by play dataset](https://www.kaggle.com/maxhorowitz/nflplaybyplay2009to2016) of NFL plays from recent years. The dataset contains roughly 100 columns of data of varying levels of importance.

### Unsupervised Approaches
For our unsupervised approach, we plan on running GMM. Ideally we will find clusters for gaining and not gaining yards. We will also probably have to do some dimensionality reduction.

### Supervised Approaches
We plan on applying a Neural Net, Decision Tree/Random Forest, and Linear Regression techniques. We also plan on using PCA for dimensionality reduction as well. From there, we will compare the performance of the six approaches and pick the best one. We will select a portion of the data as a training set and the rest will be used for testing. 

### Results
We would like to predict the amount of yard gain/loss based on a play's stats.

### Discussion
The best reasonable outcome would be for us to model 90% of plays successfully, where a successful model is defined as one that we get within 10% of the yards gained or lost.

### References
List containing at least 3 references, preferably peer reviewed (TURN TO MLA or APA)
https://towardsdatascience.com/applying-machine-learning-to-predict-byu-football-play-success-60b57267b78c  
https://dspace.mit.edu/bitstream/handle/1721.1/113120/1016455954-MIT.pdf?sequence=1  

Other dates:
Midterm report (November 6)
Final report (Demeber 7)
