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
Implements a basic perceptron neural network call.
*/
package main

import (
	"fmt"
	algo "github.com/redsofa/perceptron/algo"
	version "github.com/redsofa/perceptron/version"
)

func generateTrainingData() []*algo.TrainingData {
	var trainingData []*algo.TrainingData
	var instance *algo.TrainingData

	instance = algo.NewTrainingData([]float64{0, 0}, 0)
	trainingData = append(trainingData, instance)

	instance = algo.NewTrainingData([]float64{0, 1}, 0)
	trainingData = append(trainingData, instance)

	instance = algo.NewTrainingData([]float64{1, 0}, 0)
	trainingData = append(trainingData, instance)

	instance = algo.NewTrainingData([]float64{1, 1}, 1)
	trainingData = append(trainingData, instance)

	instance = algo.NewTrainingData([]float64{1, 2}, 0)
	trainingData = append(trainingData, instance)

	instance = algo.NewTrainingData([]float64{1, 3}, 1)
	trainingData = append(trainingData, instance)

	instance = algo.NewTrainingData([]float64{1, 4}, 1)
	trainingData = append(trainingData, instance)

	return trainingData
}

func main() {

	var threshold float64 = 0.2
	var bias float64 = 1
	var lrate float64 = 0.1
	var epoch int = 200

	println("Single-layer perceptron algorithm implementation - Version :" + version.APP_VERSION)

	trainingData := generateTrainingData()
	perceptron := algo.TrainedPerceptron(trainingData, threshold, bias, lrate, epoch)

	println(fmt.Sprintf("Predict %[1]d for : (%[2]d, %[3]d)", perceptron.CalculateOutput([]float64{1, 1}), 1, 1))
	println(fmt.Sprintf("Predict %[1]d for : (%[2]d, %[3]d)", perceptron.CalculateOutput([]float64{1, 0}), 1, 0))
	println(fmt.Sprintf("Predict %[1]d for : (%[2]d, %[3]d)", perceptron.CalculateOutput([]float64{0, 0}), 0, 0))
	println(fmt.Sprintf("Predict %[1]d for : (%[2]d, %[3]d)", perceptron.CalculateOutput([]float64{1, 4}), 1, 4))
	println(fmt.Sprintf("Predict %[1]d for : (%[2]d, %[3]d)", perceptron.CalculateOutput([]float64{1, 3}), 1, 3))
	println(fmt.Sprintf("Predict %[1]d for : (%[2]d, %[3]d)", perceptron.CalculateOutput([]float64{1, 7}), 1, 7))
	println(fmt.Sprintf("Predict %[1]d for : (%[2]d, %[3]d)", perceptron.CalculateOutput([]float64{0, -1}), 0, -1))

}
