package main

import (
	"fmt"
	"log"
	"os"

	"github.com/NickSavage/micrograd-go/nn"
)

func Run(m *nn.MLP, inputs [][]float64, targets []float64) {
	outputs := []*nn.Value{}
	for _, input := range inputs {
		inputValues := make([]*nn.Value, len(input))
		for j, v := range input {
			inputValues[j] = &nn.Value{Data: v, Grad: 0.0, Children: nil, Op: "", Label: fmt.Sprintf("x%d", j)}
		}

		output := m.Call(inputValues)
		outputs = append(outputs, output[0])
	}
	loss := &nn.Value{}
	for i, output := range outputs {
		targetValue := &nn.Value{Data: targets[i], Grad: 0.0, Children: nil, Op: "", Label: "target"}
		diff := nn.Add(output, &nn.Value{Data: -targetValue.Data, Grad: 0.0})
		result := nn.Mul(diff, diff)
		loss = nn.Add(loss, result)
	}
	loss.Backward()

	for _, param := range m.Parameters() {
		param.Data -= param.Grad * 0.01
		param.Grad = 0.0
	}
}

func main() {
	var m *nn.MLP
	if _, err := os.Stat("model.json"); os.IsNotExist(err) {
		// Create new model if file doesn't exist
		m = nn.NewMLP(1, []int{3, 4, 4, 1})
	} else {
		// Load existing model
		var err error
		m, err = nn.LoadMLP("model.json")
		if err != nil {
			log.Fatalf("Failed to load model: %v", err)
		}
	}

	// Normalize inputs to a smaller range
	inputs := [][]float64{
		{0.75},  // 75/100
		{0.25},  // 25/100
		{1.0},   // 100/100
		{0.1},   // 10/100
		{0.505}, // 50.5/100
		{0.495}, // 49.5/100
		{0.8},   // 80/100
		{0.3},   // 30/100
	}

	targets := []float64{1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0}

	// Increase training iterations and add logging
	for i := 0; i < 100; i++ {
		Run(m, inputs, targets)
	}

	// Test with normalized inputs
	testInputs := [][]float64{
		{0.6}, // 60/100
		{0.4}, // 40/100
		{0.1}, // 10/100
		{0.9}, // 90/100
	}

	// Test each input
	for _, testInput := range testInputs {
		testInputValues := make([]*nn.Value, len(testInput))
		for j, v := range testInput {
			testInputValues[j] = &nn.Value{Data: v}
		}
		output := m.Call(testInputValues)
		fmt.Printf("Input: %.1f, Output: %v (Expected: %v)\n",
			testInput[0]*100, // Denormalize for display
			output,
			map[bool]string{true: "> 50", false: "<= 50"}[testInput[0]*100 > 50])
	}

	m.Save("model.json")
}
