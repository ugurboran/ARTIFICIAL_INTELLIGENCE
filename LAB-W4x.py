"""
Uğur_BORAN_Week4_Bayesian Networks of earthquake and cloudy examples.
"""

'''
*****************************************************************************************************
*****************************************************************************************************
*************************   COMP450 LAB WORK WEEK-4 (FALL 2018-19)   ********************************
*****************************************************************************************************
*****************************************************************************************************
This lab work contains different tasks to be completed during the Week-4 lab session (October 19, 2018).
First, you need to solve the given problem by-hand on the classwork sheets.
http://homes.sice.indiana.edu/classes/spring2012/csci/b553-hauserk/bayesnet.py
'''




'''
TASK-1: Bayesian Network
This task is about the implementation of the popular earthquake example.

For more information:
- https://codesachin.wordpress.com/2017/03/10/an-introduction-to-bayesian-belief-networks/
- https://dtai.cs.kuleuven.be/problog/tutorial/basic/02_bayes.html

module implements the Bayesian network shown in the text, Figure 14.2.
It is taken from the AIMA Python code.

'''



"""
TASK-4:
In this task, you are asked to implement the following code and try different parameters.

Please explain how the following code works.
You need to provide comments for the lines that are highlighted with "COMMENT HERE!"

# From AIMA code (probability.py) - Fig. 14.2 - burglary example

"""

from aima.probability import BayesNet, enumeration_ask, elimination_ask

T, F = True, False  # 1A-1.COMMENT HERE:

burglary = BayesNet([   # 1A-2.COMMENT HERE:
    ('Burglary', '', 0.001),
    ('Earthquake', '', 0.002),
    ('Alarm', 'Burglary Earthquake', {(T, T): 0.95, (T, F): 0.94, (F, T): 0.29, (F, F): 0.001}),
    ('JohnCalls', 'Alarm', {T: 0.90, F: 0.05}),
    ('MaryCalls', 'Alarm', {T: 0.70, F: 0.01})
    ])

"""
Compute all the possible conditions. Here is a sample case for you to get help.


print(enumeration_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary).show_approx())  # 1A-3.COMMENT HERE:
print(elimination_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary).show_approx())  # 1A-4.COMMENT HERE:
"""


print("Mary CALLS")
print(elimination_ask('Burglary', dict(JohnCalls=F, MaryCalls=T), burglary).show_approx())
print(elimination_ask('Earthquake', dict(JohnCalls=F, MaryCalls=T), burglary).show_approx())
print(elimination_ask('Alarm', dict(JohnCalls=F, MaryCalls=T), burglary).show_approx())

print("jOHN CALLS")
print(elimination_ask('Burglary', dict(JohnCalls=T, MaryCalls=F), burglary).show_approx())
print(elimination_ask('Earthquake', dict(JohnCalls=T, MaryCalls=F), burglary).show_approx())
print(elimination_ask('Alarm', dict(JohnCalls=T, MaryCalls=F), burglary).show_approx())

print("Mary AND JOHN CALLS")
print(elimination_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary).show_approx())
print(elimination_ask('Earthquake', dict(JohnCalls=T, MaryCalls=T), burglary).show_approx())
print(elimination_ask('Alarm', dict(JohnCalls=T, MaryCalls=T), burglary).show_approx())

print("None CALLS")
print(elimination_ask('Burglary', dict(JohnCalls=F, MaryCalls=F), burglary).show_approx())
print(elimination_ask('Earthquake', dict(JohnCalls=F, MaryCalls=F), burglary).show_approx())
print(elimination_ask('Alarm', dict(JohnCalls=F, MaryCalls=F), burglary).show_approx())


print("\nWhen Mary and John calls alarm possibility is high and burglary is higher than earthquake so it makes sense."
      "When none calls earthquake has very low possibility and alarm and burglary is impossible."
      "When Mary calls and John calls seperately the possibilities are close to each other so it makes sense.")



"""
TASK-5:
Please do the calculations with respect to the example on Udacity that is about rain, wet ground.
Calculate all of the possible outcomes.
"""


from aima.probability import BayesNet, enumeration_ask, elimination_ask

T, F = True, False  # 1A-1.COMMENT HERE:

weather = BayesNet([   # 1A-2.COMMENT HERE:
    ('Cloudy', '', 0.5),
    ('Rain', 'Cloudy', {T: 0.8, F: 0.2}),
    ('Sprinkler', 'Cloudy', {T: 0.1, F: 0.5}),
    ('WetGrass', 'Sprinkler Rain', {(T, T): 0.99, (T, F): 0.9, (F, T): 0.9, (F, F): 0.01}),
    ])

"""
Compute all the possible conditions. Here is a sample case for you to get help.


print(enumeration_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary).show_approx())  # 1A-3.COMMENT HERE:
print(elimination_ask('Burglary', dict(JohnCalls=T, MaryCalls=T), burglary).show_approx())  # 1A-4.COMMENT HERE:
"""
"""
print(elimination_ask('Rain', dict(Cloudy=T), weather).show_approx())
print(elimination_ask('WetGrass', dict(Rain=T), weather).show_approx())
print(elimination_ask('Sprinkler', dict(Cloudy=T), weather).show_approx())
print(elimination_ask('WetGrass', dict(Sprinkler=T), weather).show_approx())
"""

print(elimination_ask('Sprinkler', dict(WetGrass=T), weather).show_approx())
print(elimination_ask('Rain', dict(WetGrass=T), weather).show_approx())
print(elimination_ask('Cloudy', dict(WetGrass=T), weather).show_approx())



