package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand/v2"
	"os"
)

type Value struct {
	Data     float64
	Grad     float64
	Children []*Value
	Op       string
	Label    string
}

type Neuron struct {
	W []*Value
	B *Value
}

type Layer struct {
	Neurons []*Neuron
}

type MLP struct {
	Layers []*Layer
}

type MLPState struct {
	NumInputs   int   `json:"num_inputs"`
	LayerSizes  []int `json:"layer_sizes"`
	LayerStates [][]struct {
		Weights []float64 `json:"weights"`
		Bias    float64   `json:"bias"`
	} `json:"layer_states"`
}

func NewNeuron(numInputs int) *Neuron {
	w := make([]*Value, numInputs)
	for i := range w {
		w[i] = &Value{
			Data:     rand.Float64()*2 - 1,
			Grad:     0.0,
			Children: nil,
			Op:       "",
			Label:    fmt.Sprintf("w%d", i),
		}
	}
	b := &Value{
		Data:     rand.Float64()*2 - 1,
		Grad:     0.0,
		Children: nil,
		Op:       "",
		Label:    "b",
	}
	return &Neuron{W: w, B: b}
}

func (n *Neuron) Print() {
	fmt.Printf("Neuron(W: %v, B: %v)\n", n.W, n.B)
}

func (n *Neuron) Call(x []*Value) *Value {
	sum := n.B
	for i, w := range n.W {
		sum = Add(sum, Mul(w, x[i]))
	}
	out := Tanh(sum)
	return out
}

func (n *Neuron) Parameters() []*Value {
	return append(n.W, n.B)
}

func NewLayer(numInputs, numOutputs int) *Layer {
	neurons := make([]*Neuron, numOutputs)
	for i := range neurons {
		neurons[i] = NewNeuron(numInputs)
	}
	return &Layer{Neurons: neurons}
}

func (l *Layer) Print() {
	for _, neuron := range l.Neurons {
		neuron.Print()
	}
}

func (l *Layer) Call(x []*Value) []*Value {
	outputs := make([]*Value, len(l.Neurons))
	for i, neuron := range l.Neurons {
		outputs[i] = neuron.Call(x)
	}
	return outputs
}

func (l *Layer) Parameters() []*Value {
	params := []*Value{}
	for _, neuron := range l.Neurons {
		params = append(params, neuron.Parameters()...)
	}
	return params
}

func NewMLP(numInputs int, layerSizes []int) *MLP {
	layers := make([]*Layer, len(layerSizes))

	// current size represents the number of inputs for the next layer
	currentSize := numInputs

	for i, outputSize := range layerSizes {
		layers[i] = NewLayer(currentSize, outputSize)
		currentSize = outputSize
	}

	return &MLP{Layers: layers}
}

func (m *MLP) Print() {
	for _, layer := range m.Layers {
		layer.Print()
	}
}

func (m *MLP) Call(x []*Value) []*Value {
	for _, layer := range m.Layers {
		x = layer.Call(x)
	}
	return x
}

func (m *MLP) Parameters() []*Value {
	params := []*Value{}
	for _, layer := range m.Layers {
		params = append(params, layer.Parameters()...)
	}
	return params
}

func (v *Value) PrintGraph(prefix string, isLast bool) {
	// First line prefix
	firstPrefix := prefix
	if isLast {
		firstPrefix += "└── "
	} else {
		firstPrefix += "├── "
	}

	// Child prefix
	childPrefix := prefix
	if isLast {
		childPrefix += "    "
	} else {
		childPrefix += "│   "
	}

	// Print current node
	opStr := ""
	if v.Op != "" {
		opStr = fmt.Sprintf(" (%s)", v.Op)
	}
	fmt.Printf("%sValue(%s: %.4f, grad=%.4f)%s\n", firstPrefix, v.Label, v.Data, v.Grad, opStr)

	// Print children
	for i, child := range v.Children {
		isLastChild := i == len(v.Children)-1
		child.PrintGraph(childPrefix, isLastChild)
	}
}

func (v *Value) Print() {
	v.PrintGraph("", true)
}

func (v *Value) String() string {
	return fmt.Sprintf("Value(Data: %v, Grad: %v)", v.Data, v.Grad)
}

func Add(a, b *Value) *Value {
	return &Value{Data: a.Data + b.Data, Grad: 0.0, Children: []*Value{a, b}, Op: "+"}
}

func Mul(a, b *Value) *Value {
	return &Value{Data: a.Data * b.Data, Grad: 0.0, Children: []*Value{a, b}, Op: "*"}
}

func Tanh(a *Value) *Value {
	return &Value{Data: math.Tanh(a.Data), Grad: 0.0, Children: []*Value{a}, Op: "tanh"}
}

func (v *Value) backward() {
	switch v.Op {
	case "+":
		v.Children[0].Grad += v.Grad
		v.Children[1].Grad += v.Grad
	case "*":
		v.Children[0].Grad += v.Grad * v.Children[1].Data
		v.Children[1].Grad += v.Grad * v.Children[0].Data
	case "tanh":
		v.Children[0].Grad += v.Grad * (1 - math.Pow(v.Data, 2))
	}
}

func (v *Value) Backward() {
	// Build nodes in topological order
	var topo []*Value
	visited := make(map[*Value]bool)

	var buildTopo func(v *Value)
	buildTopo = func(v *Value) {
		if visited[v] {
			return
		}
		visited[v] = true
		for _, child := range v.Children {
			buildTopo(child)
		}
		topo = append(topo, v)
	}
	buildTopo(v)

	// Initialize gradient of root to 1.0
	v.Grad = 1.0

	// Backpropagate in reverse topological order
	for i := len(topo) - 1; i >= 0; i-- {
		topo[i].backward()
	}
}

func (m *MLP) Save(filename string) error {
	// Create state object
	state := MLPState{
		LayerSizes: make([]int, len(m.Layers)),
		LayerStates: make([][]struct {
			Weights []float64 `json:"weights"`
			Bias    float64   `json:"bias"`
		}, len(m.Layers)),
	}

	// Get number of inputs from first layer's first neuron
	if len(m.Layers) > 0 && len(m.Layers[0].Neurons) > 0 {
		state.NumInputs = len(m.Layers[0].Neurons[0].W)
	}

	// Save each layer's state
	for i, layer := range m.Layers {
		state.LayerSizes[i] = len(layer.Neurons)
		state.LayerStates[i] = make([]struct {
			Weights []float64 `json:"weights"`
			Bias    float64   `json:"bias"`
		}, len(layer.Neurons))

		for j, neuron := range layer.Neurons {
			weights := make([]float64, len(neuron.W))
			for k, w := range neuron.W {
				weights[k] = w.Data
			}
			state.LayerStates[i][j] = struct {
				Weights []float64 `json:"weights"`
				Bias    float64   `json:"bias"`
			}{
				Weights: weights,
				Bias:    neuron.B.Data,
			}
		}
	}

	// Marshal to JSON
	data, err := json.MarshalIndent(state, "", "  ")
	if err != nil {
		return fmt.Errorf("error marshaling model state: %v", err)
	}

	// Write to file
	err = os.WriteFile(filename, data, 0644)
	if err != nil {
		return fmt.Errorf("error writing model state: %v", err)
	}

	return nil
}

func LoadMLP(filename string) (*MLP, error) {
	// Read file
	data, err := os.ReadFile(filename)
	if err != nil {
		return nil, fmt.Errorf("error reading model state: %v", err)
	}

	// Unmarshal JSON
	var state MLPState
	err = json.Unmarshal(data, &state)
	if err != nil {
		return nil, fmt.Errorf("error unmarshaling model state: %v", err)
	}

	// Create new MLP
	mlp := NewMLP(state.NumInputs, state.LayerSizes)

	// Load weights and biases
	for i, layer := range mlp.Layers {
		for j, neuron := range layer.Neurons {
			for k, w := range neuron.W {
				w.Data = state.LayerStates[i][j].Weights[k]
			}
			neuron.B.Data = state.LayerStates[i][j].Bias
		}
	}

	return mlp, nil
}

func Run(m *MLP, inputs [][]float64, targets []float64) {
	outputs := []*Value{}
	for i, input := range inputs {
		// Convert input to Values
		inputValues := make([]*Value, len(input))
		for j, v := range input {
			inputValues[j] = &Value{Data: v, Grad: 0.0, Children: nil, Op: "", Label: fmt.Sprintf("x%d", j)}
		}

		output := m.Call(inputValues)
		fmt.Printf("Input %d: %v -> Output: %v\n", i+1, input, output)
		outputs = append(outputs, output[0])
	}
	loss := &Value{}
	for i, output := range outputs {
		fmt.Printf("Output %d: %v\n", i+1, output)
		targetValue := &Value{Data: targets[i], Grad: 0.0, Children: nil, Op: "", Label: "target"}
		diff := Add(output, &Value{Data: -targetValue.Data, Grad: 0.0})
		result := Mul(diff, diff)
		loss = Add(loss, result)
	}
	fmt.Printf("Loss: %v\n", loss.Data)
	loss.Backward()
	log.Printf("%v", m.Layers[0].Neurons[0].W[0])
	// log.Printf("%v", m.Parameters())
	for _, param := range m.Parameters() {
		log.Printf("%v", param)
		param.Data -= param.Grad * 0.01
		param.Grad = 0.0
	}
}

func main() {
	var m *MLP
	if _, err := os.Stat("model.json"); os.IsNotExist(err) {
		// Create new model if file doesn't exist
		m = NewMLP(1, []int{3, 4, 4, 1})
	} else {
		// Load existing model
		var err error
		m, err = LoadMLP("model.json")
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
		testInputValues := make([]*Value, len(testInput))
		for j, v := range testInput {
			testInputValues[j] = &Value{Data: v}
		}
		output := m.Call(testInputValues)
		fmt.Printf("Input: %.1f, Output: %v (Expected: %v)\n",
			testInput[0]*100, // Denormalize for display
			output,
			map[bool]string{true: "> 50", false: "<= 50"}[testInput[0]*100 > 50])
	}

	m.Save("model.json")
}
