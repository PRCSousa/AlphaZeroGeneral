# AlphaZeroGeneral

## Generalized AlphaZero for Learning and Playing Games

## Overview

AlphaZeroGeneral is a project developed as part of the "Laboratórios de IACD" course. The primary objective of this project is to create a general AlphaZero implementation that can generalize and learn to play any game. In particular, we have tested the implementation with the games of Go and Attaxx.

## Features

- Generalized AlphaZero architecture.
- Learning and playing games knowing only the rules.
- Support for different games given correct architecture.
- Training and evaluation pipelines.

## Getting Started

### Usage
The project includes several scripts and modules for training and playing games. The main entry point is main.py, which allows you to select the game, load any previous model or create a new one, and then test or train it. Every model will be saved in AlphaZero/Models, in the folder of the alias chosen for the model.

## Games Supported
Currently, AlphaZeroGeneral has been tested with the following games:

Go
Attaxx

You can extend the project to support additional games by implementing the necessary interfaces and rules for the new game.