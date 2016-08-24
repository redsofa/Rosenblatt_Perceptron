/*Copyright 2016 Rene Richard

This file is part of Rosenblatt_Perceptron.

Rosenblatt_Perceptron is free software: you can redistribute it and/or modify
it under the terms of the Apache License as published by the Apache Software
Foundation, either version 2.0 of the License, or (at your option) any later
version.

Rosenblatt_Perceptron is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
the Apache License for more details.

You should have received a copy of the the Apache License
along with Rosenblatt_Perceptron.
If not, please see <http://www.apache.org/licenses/>.
*/

/*
Implements a basic perceptron neural network call - Earliest network model proposed
by Rosenblatt in late 1950s.
*/
package algo

import (
	"math"
	"math/rand"
	"time"
)

var EPSILON float64 = 0.00000001

//Soure :
//https://randomascii.wordpress.com/2012/02/25/comparing-floating-point-numbers-2012-edition/
func FloatEquals(a, b float64) bool {

	// Calculate the difference.
	var diff float64 = math.Abs(a - b)
	a = math.Abs(a)
	b = math.Abs(b)
	// Find the largest
	var largest float64

	if b > a {
		largest = b
	} else {
		largest = a
	}

	if diff <= largest*EPSILON {
		return true
	}
	return false
}

/* Perceptron Type */

type Perceptron struct {
	Weights      []float64 //The input weights
	Threshold    float64
	Bias         float64
	LearningRate float64
	Epoch        int
}

func TrainedPerceptron(
	trainingData []*TrainingData,
	threshold float64,
	bias float64,
	lrate float64,
	epoch int) *Perceptron {

	p := &Perceptron{}
	trDataLen := len(trainingData)
	numberOfWeights := len(trainingData[0].Inputs) + 1 //The added slot is for the bias

	r := rand.New(rand.NewSource(time.Now().UnixNano()))

	var weights = make([]float64, numberOfWeights, numberOfWeights)

	//Initialize weights to random numbers
	for i := 0; i < numberOfWeights; i++ {
		//r.Float64 return psudo-random number between  [0.0,1.0]
		weights[i] = r.Float64()
	}

	p.Weights = weights
	//Set the rest of the properties
	p.Bias = bias
	p.Threshold = threshold
	p.LearningRate = lrate
	p.Epoch = epoch

	//Train the model.
	//Cycle through the training data a set amount of times (epoch times)
	//One cycle => one epoch
	for i := 0; i < epoch; i++ {
		var globalError int64
		globalError = 0

		for j := 0; j < trDataLen; j++ {
			var estimatedOutput int64 = p.CalculateOutput(trainingData[j].Inputs)
			var expectdOutput int64 = trainingData[j].Output
			var localError int64 = expectdOutput - estimatedOutput

			globalError += localError

			//Update weights
			for k := 0; k < numberOfWeights-1; k++ {
				var delta float64
				delta = lrate * trainingData[j].Inputs[k] * float64(localError)
				p.Weights[k] += delta
			}
			//Update bias' weight
			p.Weights[numberOfWeights-1] += lrate * float64(localError)
		}
		if globalError == 0 {
			break
		}
	}

	return p
}

//Multiplies each input by its corresponding weight.
func (p *Perceptron) CalculateOutput(inputs []float64) int64 {
	sum := 0.0

	//Sum inputs * weights
	for i, input := range inputs {
		sum += input * p.Weights[i]
	}
	//Add Bias
	sum += p.Bias * p.Weights[len(p.Weights)-1]

	//Activation
	if FloatEquals(sum, p.Threshold) {
		return 1
	} else if sum > p.Threshold {
		return 1
	} else {
		return 0
	}
}

/* TrainingData Type */

type TrainingData struct {
	Inputs []float64
	Output int64
}

func NewTrainingData(inputs []float64, output int64) *TrainingData {
	return &TrainingData{
		Inputs: inputs,
		Output: output,
	}
}
