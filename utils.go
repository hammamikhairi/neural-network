package main

import (
	"encoding/json"
	"fmt"
	"math"
)

func displayProgress(number, total int) {
	progress := (float64(number) / float64(total)) * 100

	fmt.Print("\033[2K\r")
	fmt.Printf("[INFO] Epoch Progress : %.2f%%", progress)

	if number == total-1 {
		fmt.Print("\033[2K\r")
	}
}

func MaxValueIndex(outputs []float64) int {
	maxValue := math.Inf(-1)
	index := 0

	for i, val := range outputs {
		if val > maxValue {
			maxValue = val
			index = i
		}
	}

	return index
}

// helped with debugging
func d(v any) {
	b, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		panic(err)
	}
	fmt.Print(string(b))
}

func fd(a any) {
	fmt.Printf("%+v\n", a)
}

func assert(cond bool, msg string) {
	if !cond {
		panic(msg)
	}
}
