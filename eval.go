package main

import (
	"encoding/json"
	"fmt"
	"io/ioutil"
)

type History struct {
	Loss []float64
	Acc  []float64
}

type EvaluationData struct {
	numCorrect         int
	total              int
	numCorrectPerClass []int
	totalPerClass      []int
	wronglyPredictedAs []int
}

func NewEvaluationData(numClasses int) *EvaluationData {
	return &EvaluationData{
		numCorrectPerClass: make([]int, numClasses),
		totalPerClass:      make([]int, numClasses),
		wronglyPredictedAs: make([]int, numClasses),
	}
}

func (ed *EvaluationData) GetAccuracyString() string {
	return fmt.Sprintf("Predicted %d / %d (%.4f%%)", ed.numCorrect, ed.total, ed.GettAccuracy())
}

func (ed *EvaluationData) GettAccuracy() float64 {
	return (float64(ed.numCorrect) / float64(ed.total)) * 100
}

func (h *History) Save(path string) {
	jsonData, err := json.Marshal(h)
	if err != nil {
		panic(err)
	}

	// Save to a file
	err = ioutil.WriteFile(path, jsonData, 0644)
	if err != nil {
		panic(err)
	}
}

func Save(data any, path string) {
	jsonData, err := json.Marshal(data)
	if err != nil {
		panic(err)
	}

	// Save to a file
	err = ioutil.WriteFile(path, jsonData, 0644)
	if err != nil {
		panic(err)
	}
}
