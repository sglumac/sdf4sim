# Introduction and Goals

The main goal of this repository is to demonstrate the results of research conducted in [(Glumac 2022)](#glumac2022). The research introduced the use of a synchronous data flow ([(Lee 1987)](#lee1987)) as a computational model for non-iterative co-simulation masters [(Kübler 2000)](#kubler2000). The analysis of the co-simulation quality is based on the numerical defect analysis [(Enright 2000)](#enright2000). Co-simulation quality estimates are used to automatically configure a co-simulation network.

## Requirements Overview

A co-simulation network consists of Functional Mock-up Units [(blochwitz2000)](#blochwitz2000). The repository demonstrates how to automatically configure and simulate a co-simulation network for a given tolerance.

## Quality Goals

1. Verifiability (of the conclusions of the thesis) - The conclusions of the thesis should be easily verified with the code in the repository.
2. Extensibility (for the future research) - There should be an easy way to test additional hypotheses related to the research described.


The code is **not designed for**
* Performance - The code is intended to verify the research results. The target model of the calculation is fixed and the results should be the same for less CPU intensive code. This decision allows the use of Python.


# System Scope and Context

## Business Context {#_business_context}

![alt text](http://www.plantuml.com/plantuml/png/SoWkIImgAStDuOfsoomgBb58piyjoCzBpIk9vOfspCiloKWjGX9JI2nMS0RoZBZWOc2-CH8_sYUnk2Gc3rEJ2PT3QbuAq2u0 "Logo Title Text 1")

**\<optionally: Explanation of external domain interfaces>**

## Technical Context {#_technical_context}

**\<Diagram or Table>**

**\<optionally: Explanation of technical interfaces>**

**\<Mapping Input/Output to Channels>**

# Solution Strategy {#section-solution-strategy}

# Building Block View {#section-building-block-view}

## Whitebox Overall System {#_whitebox_overall_system}

***\<Overview Diagram>***

Motivation

:   *\<text explanation>*

Contained Building Blocks

:   *\<Description of contained building block (black boxes)>*

Important Interfaces

:   *\<Description of important interfaces>*

### \<Name black box 1> {#__name_black_box_1}

*\<Purpose/Responsibility>*

*\<Interface(s)>*

*\<(Optional) Quality/Performance Characteristics>*

*\<(Optional) Directory/File Location>*

*\<(Optional) Fulfilled Requirements>*

*\<(optional) Open Issues/Problems/Risks>*

### \<Name black box 2> {#__name_black_box_2}

*\<black box template>*

### \<Name black box n> {#__name_black_box_n}

*\<black box template>*

### \<Name interface 1> {#__name_interface_1}

...

### \<Name interface m> {#__name_interface_m}

## Level 2 {#_level_2}

### White Box *\<building block 1>* {#_white_box_emphasis_building_block_1_emphasis}

*\<white box template>*

### White Box *\<building block 2>* {#_white_box_emphasis_building_block_2_emphasis}

*\<white box template>*

...

### White Box *\<building block m>* {#_white_box_emphasis_building_block_m_emphasis}

*\<white box template>*

## Level 3 {#_level_3}

### White Box \<\_building block x.1\_\> {#_white_box_building_block_x_1}

*\<white box template>*

### White Box \<\_building block x.2\_\> {#_white_box_building_block_x_2}

*\<white box template>*

### White Box \<\_building block y.1\_\> {#_white_box_building_block_y_1}

*\<white box template>*

# Runtime View {#section-runtime-view}

## \<Runtime Scenario 1> {#__runtime_scenario_1}

-   *\<insert runtime diagram or textual description of the scenario>*

-   *\<insert description of the notable aspects of the interactions
    between the building block instances depicted in this diagram.\>*

## \<Runtime Scenario 2> {#__runtime_scenario_2}

## ... {#_}

## \<Runtime Scenario n> {#__runtime_scenario_n}

# Deployment View {#section-deployment-view}

## Infrastructure Level 1 {#_infrastructure_level_1}

***\<Overview Diagram>***

Motivation

:   *\<explanation in text form>*

Quality and/or Performance Features

:   *\<explanation in text form>*

Mapping of Building Blocks to Infrastructure

:   *\<description of the mapping>*

## Infrastructure Level 2 {#_infrastructure_level_2}

### *\<Infrastructure Element 1>* {#__emphasis_infrastructure_element_1_emphasis}

*\<diagram + explanation>*

### *\<Infrastructure Element 2>* {#__emphasis_infrastructure_element_2_emphasis}

*\<diagram + explanation>*

...

### *\<Infrastructure Element n>* {#__emphasis_infrastructure_element_n_emphasis}

*\<diagram + explanation>*

# Cross-cutting Concepts {#section-concepts}

## *\<Concept 1>* {#__emphasis_concept_1_emphasis}

*\<explanation>*

## *\<Concept 2>* {#__emphasis_concept_2_emphasis}

*\<explanation>*

...

## *\<Concept n>* {#__emphasis_concept_n_emphasis}

*\<explanation>*

# Architecture Decisions {#section-design-decisions}

# Quality Requirements {#section-quality-scenarios}

## Quality Tree {#_quality_tree}

## Quality Scenarios {#_quality_scenarios}

# Risks and Technical Debts {#section-technical-risks}

# Glossary {#section-glossary}

+-----------------------+-----------------------------------------------+
| Term                  | Definition                                    |
+=======================+===============================================+
| *\<Term-1>*           | *\<definition-1>*                             |
+-----------------------+-----------------------------------------------+
| *\<Term-2>*           | *\<definition-2>*                             |
+-----------------------+-----------------------------------------------+

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