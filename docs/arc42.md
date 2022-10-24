# Introduction and Goals

The main goal of this repository is to demonstrate the results of research conducted in [(Glumac 2022)](#glumac2022). The research introduced the use of a synchronous data flow ([(Lee 1987)](#lee1987)) as a computational model for non-iterative co-simulation masters [(Kübler 2000)](#kubler2000). The analysis of the co-simulation quality is based on the numerical defect analysis [(Enright 2000)](#enright2000). Co-simulation quality estimates are used to automatically configure a co-simulation network.

## Requirements Overview

A co-simulation network consists of Functional Mock-up Units [(blochwitz2000)](#blochwitz2000). The repository demonstrates how to automatically configure and simulate a co-simulation network for a given tolerance with an algorithm presented in [(Glumac 2022)](#glumac2022).

## Quality Goals

1. Verifiability (of the conclusions of the thesis) - The conclusions of the thesis should be easily verified with the code in the repository.
2. Extensibility (for the future research) - There should be an easy way to test additional hypotheses related to the research described.


The code is **not designed for**
* Performance - The code is intended to verify the research results. The target model of the calculation is fixed and the results should be the same for less CPU intensive code. This decision allows the use of Python.


# System Scope and Context

## Business Context

<img src="http://www.plantuml.com/plantuml/svg/hP6_Rjj03CPtFGNhoG5iQSdKAO2BJ0rWnCrSC7UKTEZz4KWUdFJfIobjXTWekcGZZ-J77x-x4qNMccBJHEeLj027OLJmWSbfABbfFeNiOlK4e-eadxkEyTmEGSVwN8NObQoKjNKbTLF4FAljHNqCpztXTd-SdvuUldO9HOdjwVl19Z1audPQDX-9fUIjaoEZxNIqWysXx4yXrOWp0xmLvf1uOzLpgT5333ze9lO54FeQuojWrP9GoOChUra53B8FG-KbszmqpUdLQD9MV7zhRIRvEvhLluq4ijJ_pf40PW_qGgwg9RgqP3TooQKAU5H3YEKCszEN-nle2x_rJEH27wvuw6KY3ibyMtIPj78M44nJj6O9GyOehP6kxZoI4B8RhzpvK_WeRnufllkCrQJxXoUnV9rLks5dUo8nPaSsRSEhg---zEAu7PZ3asK_oM9p3GR6QLm1rdl_2_0FfKkOLV4IPjvwHzdN57y1" width=75% height=75%>

## Technical Context

The software is written as the Python package sdf4sim available from [PyPI](https://pypi.org/project/sdf4sim/).

# Building Block View

## Class Relationship Diagram

<img src="http://www.plantuml.com/plantuml/svg/VP5DSi8m38NtEeKk06UWPT5T5y05F45fC_Y7iMGwK-AxTr9rWeWdtJb-pv_UKtOiBai97ZhlcB7ZxT6x6-4T4FjGe0BmkT--V50KhEDcCuuuNRnbT_tybLvBtrF6cXOQJAmX5EyaPHpdfygQr8CJsgTueooK6yCSvK-nE32JJ-b2aR7EhdrYzGqfmk9ROVSds7yUeUjn-2OmvpRFEc9PierjQkizmBn25PjEwZXH3aFKQGPo82vUs41gAshS6AcppvkmJowpu7UIhvGlZp_KIe_ZMeaTnNC9_WS0" width=50% height=50%>

The diagram shows the relationships between classes in *sdf.cs* and *sdf4sim.sdf*.

### Cosimulation

<img src="cosimulation.png" width=50% height=50%>

### Results

<img src="results.png" width=50% height=50%>

# Runtime View

## *sdf4sim.sdf.sequential_run*

<img src="sdf.png" width=50% height=50%>

The procedure calculates a sequential schedule for the SDF graph and runs it. It is used to implement
* *sdf4sim.cs.execute_until*
* and subsequently *sdf4sim.cs.execute*

## *sdf4sim.sdf.find_configuration*

<img src="http://www.plantuml.com/plantuml/svg/ROwn2W8n40NxVCMMMdZfFP3xax0aBxgOIq9s3y6lzvGhxQOOc5dKUDYs5lS3Dz31veYkf2bFl5cBhMQvJrbg2bB5X4kmzaJL-PyNkSIji26eh5ze8oHaH3jyJjoDrD19vGsTZoYz7b90fxExtBotLZ2uHWHb4ys2vFsyNxRk5jIqBty0">

# References

<a id="glumac2022">(Glumac 2022)</a> 
Glumac, Slaven. "Automated configuring of non-iterative co-simulation modeled by synchronous data flow." PhD diss., University of Zagreb. Faculty of Electrical Engineering and Computing. Department of Control and Computer Engineering, 2022.

<a id="lee1987">(Lee 1987)</a> 
Lee, Edward A., and David G. Messerschmitt. "Synchronous data flow." Proceedings of the IEEE 75, no. 9 (1987): 1235-1245.

<a id="kubler2000">(Kübler 2000)</a> 
Kübler, Ralf, and Werner Schiehlen. "Two methods of simulator coupling." Mathematical and computer modelling of dynamical systems 6, no. 2 (2000): 93-113.

<a id="enright2000">(Enright 2000)</a> 
Enright, W. H. "Continuous numerical methods for ODEs with defect control." Journal of computational and applied mathematics 125, no. 1-2 (2000): 159-170.

<a id="blochwitz2000">(Blochwitz 2000)</a> 
Blochwitz, Torsten, Martin Otter, Martin Arnold, Constanze Bausch, Christoph Clauß, Hilding Elmqvist, Andreas Junghanns et al. "The functional mockup interface for tool independent exchange of simulation models." In Proceedings of the 8th international Modelica conference, pp. 105-114. Linköping University Press, 2011.

# 

This documentation is based on the modification of the arc42 template. Some of the content suggested by the original template is not displayed.

**About arc42**

arc42, the template for documentation of software and system
architecture.

Template Version 8.1 EN. (based upon AsciiDoc version), May 2022

Created, maintained and © by Dr. Peter Hruschka, Dr. Gernot Starke and
contributors. See <https://arc42.org>.
